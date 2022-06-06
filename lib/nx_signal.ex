defmodule NxSignal do
  @moduledoc """
  Nx library extension for DSP
  """

  import Nx.Defn

  @doc """
  Computes the Short-Time Fourier Transform of a tensor.

  Returns the complex spectrum Z, the time in seconds for
  each frame and the frequency bins in Hz.

  See also: `NxSignal.Windows`

  ## Options

    * `:fs` - the sampling frequency for the input in Hz. Defaults to `1000`.
    * `:nfft` - the DFT length that will be passed to `Nx.fft/2`. Defaults to `:power_of_two`.
    * `overlap_size` - the number of samples for the overlap between frames.
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
      |> as_windowed(window_dimensions: {frame_size}, strides: [frame_size - overlap_size])
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
  Returns a tensor with rank N+1 in which the last N dimensions
  are sliding windows of rank N over the input tensor of rank N.

  ## Options

    * `:strides` - a list of length N which represents the strides
      over each dimension. Can also be a number which will be the
      stride over each dimension. Defaults to `1`.
    * `:padding` - A valid padding as per `Nx.Shape.pad/2` over the
      input tensor's shape. Defaults to `:valid`

  ## Examples

      iex> NxSignal.as_windowed(Nx.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 11, 10]]), window_dimensions: {2, 4})
      #Nx.Tensor<
        s64[2][2][4]
        [
          [
            [0, 1, 2, 3],
            [10, 11, 12, 11]
          ],
          [
            [1, 2, 3, 4],
            [11, 12, 11, 10]
          ]
        ]
      >

      iex> NxSignal.as_windowed(Nx.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 11, 10]]), window_dimensions: {2, 3})
      #Nx.Tensor<
        s64[3][2][3]
        [
          [
            [0, 1, 2],
            [10, 11, 12]
          ],
          [
            [1, 2, 3],
            [11, 12, 11]
          ],
          [
            [2, 3, 4],
            [12, 11, 10]
          ]
        ]
      >

      iex> t = Nx.tensor([
      ...>  [0, 1, 2],
      ...>  [3, 4, 5],
      ...>  [-1, -2, -3],
      ...>  [-4, -5, -6],
      ...>  [-7, -8, -9]
      ...> ])
      iex> NxSignal.as_windowed(t, window_dimensions: {3, 2}, strides: [2, 1])
      #Nx.Tensor<
        s64[4][3][2]
        [
          [
            [0, 1],
            [3, 4],
            [-1, -2]
          ],
          [
            [1, 2],
            [4, 5],
            [-2, -3]
          ],
          [
            [-1, -2],
            [-4, -5],
            [-7, -8]
          ],
          [
            [-2, -3],
            [-5, -6],
            [-8, -9]
          ]
        ]
      >
  """
  defn as_windowed(tensor, opts \\ []) do
    {window_dimensions, window_min_source_shape, strides, padding, window_dimensions_list,
     output_shape} =
      transform({Nx.shape(tensor), opts}, fn {shape, opts} ->
        opts = Keyword.validate!(opts, [:window_dimensions, padding: :valid, strides: 1])
        window_dimensions = opts[:window_dimensions]

        padding = opts[:padding]

        strides =
          case opts[:strides] do
            strides when Elixir.Kernel.is_list(strides) ->
              strides

            strides
            when Elixir.Kernel.and(
                   Elixir.Kernel.is_integer(strides),
                   Elixir.Kernel.>=(strides, 1)
                 ) ->
              List.duplicate(strides, Nx.rank(tensor))

            strides ->
              raise ArgumentError,
                    "expected an integer >= 1 or a list of integers, got: #{inspect(strides)}"
          end

        dilations = List.duplicate(1, Nx.rank(tensor))

        {window_min_source_shape, _} =
          Nx.Shape.pool(shape, window_dimensions, strides, padding, dilations)

        window_dimensions_list = Tuple.to_list(window_dimensions)

        output_shape =
          List.to_tuple([Tuple.product(window_min_source_shape) | window_dimensions_list])

        {window_dimensions, window_min_source_shape, strides, padding, window_dimensions_list,
         output_shape}
      end)

    window_min_source = Nx.broadcast(1, window_min_source_shape)

    t_iota = Nx.iota(tensor)

    # Scatter the min index for each window.
    # Since strides is always != 0, we don't
    # have any overlapping scatters, which
    # results in a tensor which marks the start
    # of each window.
    scattered_indices =
      t_iota
      |> Nx.window_scatter_min(window_min_source, 0, window_dimensions,
        strides: strides,
        padding: padding
      )
      |> then(&Nx.select(&1, t_iota, -1))

    {sliced_indices, template_indices} =
      transform(
        {t_iota, window_dimensions_list, scattered_indices, window_min_source_shape, strides},
        &slice_indices/1
      )

    # Tile the sliced indices in way that
    # each index will be the start of a 'row'
    # of length Tuple.product(window_dimensions).
    # This 'row' will be used to offset a template
    # index tensor which indexes the first window.
    # The offset indices represent each window in the
    # flat tensor.
    # These indices can be used in Nx.take(Nx.flatten(tensor), idx)
    # Which we then need to reshape to obtain the correctly shaped windows
    # in the last N dimensions.
    idx =
      sliced_indices
      |> Nx.flatten()
      |> Nx.new_axis(1)
      |> Nx.tile([1, Nx.size(template_indices)])
      |> Nx.add(template_indices)
      |> Nx.reshape(output_shape)

    tensor
    |> Nx.flatten()
    |> Nx.take_along_axis(Nx.flatten(idx))
    |> Nx.reshape(idx.shape)
  end

  defp slice_indices(
         {t_iota, window_dimensions_list, scattered_indices, window_min_source_shape, strides}
       ) do
    # Since we can have windows which aren't touching,
    # The can be "-1" gaps in the scattered_indices tensor
    # above. For example:
    # [
    #   [0, 1, -1],
    #   [-1, -1, -1],
    #   [6, 7, -1],
    #   [-1, -1, -1],
    #   [-1, -1, -1]
    # ]
    # Because of this, we need to sort the tensor in all dimensions
    # where the strides are bigger than 1. After sorting where needed,
    # We need to slice the tensor to get all window starts in which we
    # are interested.

    zeros = List.duplicate(0, Nx.rank(scattered_indices))

    sliced_indices =
      for {stride, axis} <- Enum.with_index(strides), reduce: scattered_indices do
        idx ->
          len = elem(window_min_source_shape, axis)
          lengths = idx.shape |> Tuple.to_list() |> List.replace_at(axis, len)

          if stride > 1 do
            sorted = Nx.sort(idx, axis: axis, direction: :asc)
            starts = List.replace_at(zeros, axis, elem(idx.shape, axis) - len)

            Nx.slice(sorted, starts, lengths)
          else
            # If we're not sorting, it means that the positions
            # we're interested in are at the start.
            # Therefore, the slicing needs to take place from the
            # start of the tensor
            Nx.slice(idx, zeros, lengths)
          end
      end

    # Template indices which index the first window in the tensor
    template_indices =
      t_iota
      |> Nx.slice(List.duplicate(0, Nx.rank(t_iota)), window_dimensions_list)
      |> Nx.flatten()

    {sliced_indices, template_indices}
  end
end
