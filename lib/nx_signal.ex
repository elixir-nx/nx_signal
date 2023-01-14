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
    * `:window_padding` - `:reflect`, `:zeros` or `nil`. See `as_windowed/3` for more details.

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
        f32[frequencies: 2]
        [0.0, 200.0]
      >
  """
  defn stft(data, window, opts \\ []) do
    {frame_size} = Nx.shape(window)

    {overlap_size, fs, padding} =
      transform({frame_size, opts}, fn {frame_size, opts} ->
        opts =
          Keyword.validate!(opts, [
            :overlap_size,
            :window,
            window_padding: :valid,
            fs: 100,
            nfft: :power_of_two
          ])

        fs = opts[:fs] || raise ArgumentError, "missing fs option"

        overlap_size = opts[:overlap_size] || div(frame_size, 2)

        {overlap_size, fs, opts[:window_padding]}
      end)

    spectrum =
      data
      |> as_windowed(
        padding: padding,
        window_size: frame_size,
        stride: frame_size - overlap_size
      )
      |> Nx.multiply(window)
      |> Nx.fft(length: opts[:nfft])

    {num_frames, num_frequencies} = Nx.shape(spectrum)

    frequencies =
      stft_frequencies(
        fs: fs,
        num_frequencies: num_frequencies
      )

    # assign the middle of the equivalent time window as the time for the given frame
    times = (Nx.iota({num_frames}, names: [:frames]) + 1) * frame_size / (2 * fs)

    {Nx.reshape(spectrum, spectrum.shape, names: [:frames, :frequencies]), times, frequencies}
  end

  defnp stft_frequencies(opts \\ []) do
    opts = keyword!(opts, [:num_frequencies, :fs, type: {:f, 32}, names: [:frequencies]])

    Nx.iota({opts[:num_frequencies]}, type: opts[:type], names: opts[:names]) * opts[:fs] /
      opts[:num_frequencies]
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
    * `:padding` - A can be `:reflect` or a  valid padding as per `Nx.Shape.pad/2` over the
      input tensor's shape. Defaults to `:valid`. If `:reflect` or `:zeros`, the first window will be centered
      at the start of the signal. For `:reflect`, each incomplete window will be reflected as if it was
      periodic (see examples for `as_windowed/2`). For `:zeros`, each incomplete window will be zero-padded.

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

      iex> t = Nx.iota({7});
      iex> NxSignal.as_windowed(t, window_size: 6, padding: :reflect, stride: 1)
      #Nx.Tensor<
        s64[8][6]
        [
          [1, 2, 1, 0, 1, 2],
          [2, 1, 0, 1, 2, 3],
          [1, 0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5, 6],
          [2, 3, 4, 5, 6, 5],
          [3, 4, 5, 6, 5, 4],
          [4, 5, 6, 5, 4, 3]
        ]
      >
  """
  deftransform as_windowed(tensor, opts \\ []) do
    if opts[:padding] == :reflect do
      as_windowed_reflect_padding(tensor, opts)
    else
      as_windowed_non_reflect_padding(tensor, opts)
    end
  end

  deftransformp as_windowed_parse_opts(shape, opts, :reflect) do
    window_size = opts[:window_size]

    as_windowed_parse_opts(
      shape,
      Keyword.put(opts, :padding, [{div(window_size, 2), div(window_size, 2)}])
    )
  end

  deftransformp as_windowed_parse_opts(shape, opts) do
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

    dilations = List.duplicate(1, Nx.rank(shape))

    {pooled_shape, padding_config} =
      Nx.Shape.pool(shape, window_dimensions, strides, padding, dilations)

    output_shape = {Tuple.product(pooled_shape), window_size}

    {window_size, stride, Enum.map(padding_config, fn {x, y} -> {x, y, 0} end), output_shape}
  end

  defnp as_windowed_non_reflect_padding(tensor, opts \\ []) do
    # current implementation only supports windowing 1D tensors
    {window_size, stride, padding, output_shape} = as_windowed_parse_opts(Nx.shape(tensor), opts)

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

  defnp as_windowed_reflect_padding(tensor, opts \\ []) do
    # current implementation only supports windowing 1D tensors
    {window_size, stride, _padding, output_shape} =
      as_windowed_parse_opts(Nx.shape(tensor), opts, :reflect)

    output = Nx.broadcast(Nx.tensor(0, type: tensor.type), output_shape)
    {num_windows, _} = Nx.shape(output)

    index_template =
      Nx.concatenate([Nx.broadcast(0, {window_size, 1}), Nx.iota({window_size, 1})], axis: 1)

    leading_window_indices = generate_leading_window_indices(window_size, stride)

    {output, _, _, _, _} =
      while {output, i = 0, current_window = 0, t = tensor, index_template},
            current_window < num_windows do
        # Here windows are centered at the current index

        cond do
          i < div(window_size, 2) ->
            # We're indexing before we have a full window on the left

            window = Nx.take(t, leading_window_indices[i])

            indices = index_template + Nx.stack([current_window, 0])
            updated = Nx.indexed_add(output, indices, window)

            {updated, i + stride, current_window + 1, t, index_template}

          true ->
            # Case where we can index a full window
            indices = index_template + Nx.stack([current_window, 0])
            updates = t |> Nx.slice([i - div(window_size, 2)], [window_size]) |> Nx.flatten()

            updated = Nx.indexed_add(output, indices, updates)

            {updated, i + stride, current_window + 1, t, index_template}
        end
      end

    # Now we need to handle the tail-end of the windows,
    # since they are currently all the same value. We want to apply the tapering-off
    # like we did with the initial windows.

    apply_right_padding(output)
  end

  deftransformp generate_leading_window_indices(window_size, stride) do
    half_window = div(window_size, 2)

    for offset <- 0..half_window//stride do
      {offset + half_window}
      |> Nx.iota()
      |> pad_reflect(window_size)
    end
    |> Nx.stack()
  end

  defnp apply_right_padding(output) do
    offsets =
      (output[-1] == output) |> Nx.all(axes: [1], keep_axes: true) |> Nx.cumulative_sum(axis: 0)

    idx = Nx.iota(Nx.shape(output), axis: 1) + Nx.select(offsets != 0, offsets - 1, 0)

    [output, Nx.reverse(output[[0..-1//1, 0..-2//1]], axes: [1])]
    |> Nx.concatenate(axis: 1)
    |> Nx.take_along_axis(idx, axis: 1)
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
                "overlap_size must be a number less than the window size #{window_size}, got: #{inspect(window_size)}"
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

  @doc """
  Generates MEL-scale weights

  See also: `stft/3`
  """
  defn mel_filters(opts \\ []) do
    opts = keyword!(opts, [:fs, :nfft, n_mels: 128, type: {:f, 32}])
    n_mels = opts[:n_mels]
    nfft = opts[:nfft]
    fs = opts[:fs]
    type = opts[:type]

    fftfreqs = stft_frequencies(fs: fs, type: type, num_frequencies: nfft)

    # magic numbers :p
    min_mel = 0
    max_mel = 45.245640471924965

    mels = linspace(min_mel, max_mel, n: n_mels)
    f_min = 0
    f_sp = 200 / 3
    freqs = f_min + f_sp * mels

    min_log_hz = 1_000
    min_log_mel = (min_log_hz - f_min) / f_sp

    logstep = Nx.log(6.4) / 27

    log_t = mels >= min_log_mel

    # This is the same as freqs[log_t] = min_log_hz * Nx.exp(logstep * (mels[log_t] - min_log_mel))
    # notice that since freqs and mels are indexed by the same conditional tensor, we don't
    # need to slice either of them
    freqs = Nx.select(log_t, min_log_hz * Nx.exp(logstep * (mels - min_log_mel)), freqs)

    mel_f = freqs

    fdiff = mel_f[1..-1//1] - mel_f[0..-2//1]
    ramps = Nx.new_axis(mel_f, 1) - fftfreqs

    lower = -ramps[0..(n_mels - 1)] / fdiff[0..(n_mels - 1)]
    upper = ramps[2..(n_mels + 1)//1] / fdiff[1..n_mels]
    weights = Nx.max(0, Nx.min(lower, upper))

    enorm = 2.0 / (mel_f[2..(n_mels + 1)] - mel_f[0..(n_mels - 1)])
    weights * Nx.new_axis(enorm, 1)
  end

  defn stft_to_mel(z, opts \\ []) do
    magnitudes = Nx.abs(z) ** 2
    filters = mel_filters(opts)

    mel_spec = Nx.dot(filters, magnitudes)
    log_spec = Nx.log(Nx.max(mel_spec, 1.0e-10)) / Nx.log(10)
    log_spec = Nx.max(log_spec, Nx.reduce_max(log_spec) - 8)
    (log_spec + 4) / 4
  end

  defnp linspace(min, max, opts \\ []) do
    [n: n] = keyword!(opts, [:n])

    step = (max - min) / n
    min + Nx.iota({n}) * step
  end

  defn pad_reflect(window, target_size) do
    case Nx.shape(window) do
      {^target_size} ->
        window

      {n} ->
        pad_length = target_size - n

        period =
          case pad_length do
            1 ->
              Nx.tensor([1])

            2 ->
              Nx.tensor([1, 2]) |> Nx.remainder(n)

            _ ->
              Nx.concatenate([
                Nx.iota({n - 1})[1..-1//1],
                (n - Nx.iota({n})) |> Nx.slice([1], [n - 1]),
                Nx.tensor([0])
              ])
          end

        idx = Nx.iota({pad_length}) |> Nx.remainder(2 * n - 2)
        pad = window |> Nx.take(Nx.take(period, idx)) |> Nx.reverse()

        Nx.concatenate([pad, window])
    end
  end
end
