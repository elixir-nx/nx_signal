defmodule NxSignal do
  @moduledoc """
  Nx library extension for DSP
  """

  import Nx.Defn

  @doc """
  Computes the Short-Time Fourier Transform of a tensor.

  Returns the complex spectrum Z, the time in seconds for
  each frame and the frequency bins in Hz.

  See also: `NxSignal.Windows`, `Nx.Signal.istft`

  ## Options

    * `:fs` - the sampling frequency for the input in Hz. Defaults to `1000`.
    * `:nfft` - the DFT length that will be passed to `Nx.fft/2`. Defaults to `:power_of_two`.
    * `:overlap_size` - the number of samples for the overlap between frames.
    Defaults to `div(frame_size, 2)`.

  ## Examples

      iex> {z, t, f} = NxSignal.stft(Nx.iota({4}), NxSignal.Windows.rectangular(n: 2), overlap_size: 1, nfft: 2, fs: 400)
      iex> z
      #Nx.Tensor<
        c64[frames: 3][frequencies: 2]
        [
          [1.0+0.0i, -1.0+0.0i],
          [3.0+0.0i, -1.0+0.0i],
          [5.0+0.0i, -1.0+0.0i]
        ]
      >
      iex> t
      #Nx.Tensor<
        f32[frames: 3]
        [0.0024999999441206455, 0.004999999888241291, 0.007499999832361937]
      >
      iex> f
      #Nx.Tensor<
        f64[frequencies: 2]
        [0.0, 200.0]
      >
  """
  defn stft(data, window, opts \\ []) do
    {frame_size} = Nx.shape(window)

    {overlap_size, fs} =
      transform({frame_size, opts}, fn {frame_size, opts} ->
        opts =
          Keyword.validate!(opts, [
            :overlap_size,
            :window,
            fs: 100,
            nfft: :power_of_two
          ])

        fs = opts[:fs] || raise ArgumentError, "missing fs option"

        overlap_size = opts[:overlap_size] || div(frame_size, 2)

        {overlap_size, fs}
      end)

    spectrum =
      data
      |> as_windowed(window_size: frame_size, stride: frame_size - overlap_size)
      |> Nx.multiply(window)
      |> Nx.fft(length: opts[:nfft])

    {num_frames, num_frequencies} = Nx.shape(spectrum)

    frequencies =
      Nx.iota({num_frequencies}, type: {:f, 64}, names: [:frequencies]) * fs / num_frequencies

    # assign the middle of the equivalent time window as the time for the given frame
    times = (Nx.iota({num_frames}, names: [:frames]) + 1) * frame_size / (2 * fs)

    {Nx.reshape(spectrum, spectrum.shape, names: [:frames, :frequencies]), times, frequencies}
  end

  @doc """
  Computes the Inverse Short-Time Fourier Transform of a tensor.

  Returns a tensor of M time-domain frames of length `nfft`.

  See also: `NxSignal.Windows`, `Nx.Signal.stft`

  ## Options

    * `:nfft` - the DFT length that will be passed to `Nx.fft/2`. Defaults to `:power_of_two`.

  ## Examples

      iex> z = Nx.tensor([
      ...>   [1, -1],
      ...>   [3, -1],
      ...>   [5, -1]
      ...> ])
      iex> NxSignal.istft(z, NxSignal.Windows.rectangular(n: 2), overlap_size: 1, nfft: 2, fs: 400)
      #Nx.Tensor<
        c64[frames: 3][samples: 2]
        [
          [0.0+0.0i, 1.0+0.0i],
          [1.0+0.0i, 2.0+0.0i],
          [2.0+0.0i, 3.0+0.0i]
        ]
      >
  """
  defn istft(data, window, opts \\ []) do
    frames =
      data
      |> Nx.ifft(length: opts[:nfft])
      |> Nx.multiply(window)

    Nx.reshape(frames, frames.shape, names: [:frames, :samples])
  end

  @doc """
  Returns a tensor of K windows of length N

  ## Options

    * `:window_size` - the number of samples in a window
    * `:stride` - The number of samples to skip between windows. Defaults to `1`.
    * `:padding` - A valid padding as per `Nx.Shape.pad/2` over the
      input tensor's shape. Defaults to `:valid`

  ## Examples

      iex> NxSignal.as_windowed(Nx.tensor([0, 1, 2, 3, 4, 10, 11, 12]), window_size: 4)
      #Nx.Tensor<
        s64[5][4]
        [
          [0, 1, 2, 3],
          [1, 2, 3, 4],
          [2, 3, 4, 10],
          [3, 4, 10, 11],
          [4, 10, 11, 12]
        ]
      >

      iex> NxSignal.as_windowed(Nx.tensor([0, 1, 2, 3, 4, 10, 11, 12]), window_size: 3)
      #Nx.Tensor<
        s64[6][3]
        [
          [0, 1, 2],
          [1, 2, 3],
          [2, 3, 4],
          [3, 4, 10],
          [4, 10, 11],
          [10, 11, 12]
        ]
      >

      iex> NxSignal.as_windowed(Nx.tensor([0, 1, 2, 3, 4, 10, 11]), window_size: 2, stride: 2, padding: [{0, 3}])
      #Nx.Tensor<
        s64[5][2]
        [
          [0, 1],
          [2, 3],
          [4, 10],
          [11, 0],
          [0, 0]
        ]
      >
  """
  defn as_windowed(tensor, opts \\ []) do
    # current implementation only supports windowing 1D tensors
    {window_size, stride, padding, output_shape} =
      transform({Nx.shape(tensor), opts}, fn {shape, opts} ->
        opts = Keyword.validate!(opts, [:window_size, padding: :valid, stride: 1])
        window_size = opts[:window_size]
        window_dimensions = {window_size}

        padding = opts[:padding]

        [stride] =
          strides =
          case opts[:stride] do
            stride when Elixir.Kernel.is_list(stride) ->
              stride

            stride
            when Elixir.Kernel.and(
                   Elixir.Kernel.is_integer(stride),
                   Elixir.Kernel.>=(stride, 1)
                 ) ->
              [stride]

            stride ->
              raise ArgumentError,
                    "expected an integer >= 1 or a list of integers, got: #{inspect(stride)}"
          end

        dilations = List.duplicate(1, Nx.rank(tensor))

        {pooled_shape, padding_config} =
          Nx.Shape.pool(shape, window_dimensions, strides, padding, dilations)

        output_shape = {Tuple.product(pooled_shape), window_size}

        {window_size, stride, Enum.map(padding_config, fn {x, y} -> {x, y, 0} end), output_shape}
      end)

    output = Nx.broadcast(Nx.tensor(0, type: tensor.type), output_shape)
    {num_windows, _} = Nx.shape(output)

    index_template =
      Nx.concatenate([Nx.broadcast(0, {window_size, 1}), Nx.iota({window_size, 1})], axis: 1)

    {output, _, _, _, _} =
      while {output, i = 0, current_window = 0, t = Nx.pad(tensor, 0, padding), index_template},
            current_window < num_windows do
        indices = index_template + Nx.stack([current_window, 0])
        updates = t |> Nx.slice([i], [window_size]) |> Nx.flatten()

        updated = Nx.indexed_add(output, indices, updates)

        {updated, i + stride, current_window + 1, t, index_template}
      end

    output
  end

  @doc """
  Performs the overlap-and-add algorithm over
  an M by N tensor, where M is the number of
  windows and N is the window size.

  The tensor is zero-padded on the right so
  the last window fully appears in the result.

  ## Options

    * `:overlap_size` - The number of overlapping samples between windows

  ## Examples

      iex> NxSignal.overlap_and_add(Nx.iota({3, 4}), overlap_size: 0)
      #Nx.Tensor<
        s64[12]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
      >

      iex> NxSignal.overlap_and_add(Nx.iota({3, 4}), overlap_size: 3)
      #Nx.Tensor<
        s64[6]
        [0, 5, 15, 18, 17, 11]
      >

      iex> NxSignal.overlap_and_add(Nx.iota({3, 4}), overlap_size: -3)
      #Nx.Tensor<
        s64[18]
        [0, 1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 0, 0, 0, 8, 9, 10, 11]
      >

  """
  defn overlap_and_add(tensor, opts \\ []) do
    {stride, num_windows, window_size, output_holder_shape} =
      transform({tensor, opts}, fn {tensor, opts} ->
        import Nx.Defn.Kernel, only: []
        import Elixir.Kernel

        {num_windows, window_size} = Nx.shape(tensor)
        overlap_size = opts[:overlap_size]

        unless is_number(overlap_size) and overlap_size < window_size do
          raise ArgumentError,
                "overlap_size must be a number greater than 0 and less than the window size #{window_size}, got: #{inspect(window_size)}"
        end

        stride = window_size - overlap_size

        output_holder_shape = {num_windows * stride + overlap_size}

        {stride, num_windows, window_size, output_holder_shape}
      end)

    {output, _, _, _, _, _} =
      while {
              out = Nx.broadcast(0, output_holder_shape),
              tensor,
              i = 0,
              idx_template = Nx.iota({window_size, 1}),
              stride,
              num_windows
            },
            i < num_windows do
        current_window = tensor[i]
        idx = idx_template + i * stride

        {
          Nx.indexed_add(out, idx, current_window),
          tensor,
          i + 1,
          idx_template,
          stride,
          num_windows
        }
      end

    output
  end
end
