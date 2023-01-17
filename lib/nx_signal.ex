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
  deftransform stft(data, window, opts \\ []) do
    {frame_size} = Nx.shape(window)

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

    stft_n(data, window, fs, Keyword.put(opts, :overlap_size, overlap_size))
  end

  defnp stft_n(data, window, fs, opts \\ []) do
    {frame_size} = Nx.shape(window)
    padding = opts[:window_padding]
    nfft = opts[:nfft]
    overlap_size = opts[:overlap_size]

    spectrum =
      data
      |> as_windowed(
        padding: padding,
        window_size: frame_size,
        stride: frame_size - overlap_size
      )
      |> Nx.multiply(window)
      |> Nx.fft(length: nfft)

    {num_frames, nfft} = Nx.shape(spectrum)

    frequencies = fft_frequencies(fs, nfft: nfft)

    # assign the middle of the equivalent time window as the time for the given frame
    time_step = frame_size / (2 * fs)
    last_frame = time_step * num_frames
    times = Nx.linspace(time_step, last_frame, n: num_frames, name: :frames)

    {Nx.reshape(spectrum, spectrum.shape, names: [:frames, :frequencies]), times, frequencies}
  end

  @doc """
  Computes the frequency bins for a FFT with given options.

  ## Arguments

    * `fs` - Sampling frequency in Hz.

  ## Options

    * `:nfft` - Number of FFT frequency bins.
    * `:type` - Optional output type. Defaults to `{:f, 32}`
    * `:name` - Optional axis name for the tensor. Defaults to `:frequencies`

  ## Examples

      iex> NxSignal.fft_frequencies(1.6e4, nfft: 10)
      #Nx.Tensor<
        f32[frequencies: 10]
        [0.0, 1.6e3, 3.2e3, 4.8e3, 6.4e3, 8.0e3, 9.6e3, 1.12e4, 1.28e4, 1.44e4]
      >
  """
  defn fft_frequencies(fs, opts \\ []) do
    opts = keyword!(opts, [:nfft, type: {:f, 32}, name: :frequencies, endpoint: false])
    nfft = opts[:nfft]

    step = fs / nfft

    Nx.linspace(0, step * nfft,
      n: nfft,
      type: opts[:type],
      name: opts[:name],
      endpoint: opts[:endpoint]
    )
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
        s64[7][6]
        [
          [1, 2, 1, 0, 1, 2],
          [2, 1, 0, 1, 2, 3],
          [1, 0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5, 6],
          [2, 3, 4, 5, 6, 5],
          [3, 4, 5, 6, 5, 4]
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
      Keyword.put(opts, :padding, [{div(window_size, 2), div(window_size, 2) - 1}])
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
        stride when is_list(stride) ->
          stride

        stride when is_integer(stride) and stride >= 1 ->
          [stride]

        stride ->
          raise ArgumentError,
                "expected an integer >= 1 or a list of integers, got: #{inspect(stride)}"
      end

    dilations = List.duplicate(1, Nx.rank(shape))

    {pooled_shape, padding_config} =
      NxSignal.Shape.pool(shape, window_dimensions, strides, padding, dilations)

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
    half_window = div(window_size - 1, 2) + 1

    {output, _, _, _, _} =
      while {output, i = 0, current_window = 0, t = tensor, index_template},
            current_window < num_windows do
        # Here windows are centered at the current index

        cond do
          i < half_window ->
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

  defnp pad_reflect(window, target_size) do
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
  Generates weights for converting an STFT representation into MEL-scale.

  See also: `stft/3`, `istft/3`, `stft_to_mel/2`

  ## Arguments

    * `nfft` - Number of FFT bins
    * `nmels` - Number of target MEL bins
    * `fs` - Sampling frequency in Hz

  ## Options
    * `:max_mel` - the pitch for the last MEL bin before log scaling. Defaults to 3016
    * `:mel_frequency_spacing` - the distance in Hz between two MEL bins before log scaling. Defaults to 66.6
    * `:type` - Target output type. Defaults to `{:f, 32}`

  ## Examples

      iex> NxSignal.mel_filters(10, 5, 8.0e3)
      #Nx.Tensor<
        f32[mels: 5][frequencies: 10]
        [
          [0.0, 8.129207999445498e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 9.972016559913754e-4, 2.1870288765057921e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 9.510891977697611e-4, 4.150509194005281e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 4.035891906823963e-4, 5.276656011119485e-4, 2.574124082457274e-4, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 7.329034269787371e-5, 2.342205698369071e-4, 3.8295105332508683e-4, 2.8712040511891246e-4, 1.9128978601656854e-4, 9.545915963826701e-5]
        ]
      >
  """
  deftransform mel_filters(nfft, nmels, fs, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        max_mel: 3016,
        mel_frequency_spacing: 200 / 3,
        type: {:f, 32}
      )

    mel_filters_n(fs, opts[:max_mel], opts[:mel_frequency_spacing],
      type: opts[:type],
      nfft: nfft,
      nmels: nmels
    )
  end

  defnp mel_filters_n(fs, max_mel, f_sp, opts \\ []) do
    nfft = opts[:nfft]
    nmels = opts[:nmels]
    type = opts[:type]

    fftfreqs = fft_frequencies(fs, type: type, nfft: nfft)

    mels = Nx.linspace(0, max_mel / f_sp, type: type, n: nmels + 2, name: :mels)
    freqs = f_sp * mels

    min_log_hz = 1_000
    min_log_mel = min_log_hz / f_sp

    # numpy uses the f64 value by default
    logstep = Nx.log(6.4) / 27

    log_t = mels >= min_log_mel

    # This is the same as freqs[log_t] = min_log_hz * Nx.exp(logstep * (mels[log_t] - min_log_mel))
    # notice that since freqs and mels are indexed by the same conditional tensor, we don't
    # need to slice either of them
    mel_f = Nx.select(log_t, min_log_hz * Nx.exp(logstep * (mels - min_log_mel)), freqs)

    fdiff = Nx.new_axis(mel_f[1..-1//1] - mel_f[0..-2//1], 1)
    ramps = Nx.new_axis(mel_f, 1) - fftfreqs

    lower = -ramps[0..(nmels - 1)] / fdiff[0..(nmels - 1)]
    upper = ramps[2..(nmels + 1)//1] / fdiff[1..nmels]
    weights = Nx.max(0, Nx.min(lower, upper))

    enorm = 2.0 / (mel_f[2..(nmels + 1)] - mel_f[0..(nmels - 1)])

    weights * Nx.new_axis(enorm, 1)
  end

  @doc """
  Converts a given STFT time-frequency spectrum into a MEL-scale time-frequency spectrum.

  See also: `stft/3`, `istft/3`, `mel_filters/1`

  ## Arguments
    * `z` - STFT spectrum
    * `fs` - Sampling frequency in Hz

  ## Options

    * `:nfft` - Number of FFT bins
    * `:nmels` - Number of target MEL bins. Defaults to 128
    * `:type` - Target output type. Defaults to `{:f, 32}`

  ## Examples

      iex> nfft = 16
      iex> fs = 8.0e3
      iex> {z, _, _} = NxSignal.stft(Nx.iota({10}), NxSignal.Windows.hann(n: 4), overlap_size: 2, nfft: nfft, fs: fs, window_padding: :reflect)
      iex> Nx.axis_size(z, :frequencies)
      16
      iex> Nx.axis_size(z, :frames)
      5
      iex> NxSignal.stft_to_mel(z, fs, nfft: nfft, nmels: 4)
      #Nx.Tensor<
        f32[frames: 5][mel: 4]
        [
          [0.2900530695915222, 0.17422175407409668, 0.18422472476959229, 0.09807997941970825],
          [0.6093881130218506, 0.5647397041320801, 0.4353824257850647, 0.08635270595550537],
          [0.7584103345870972, 0.7085014581680298, 0.5636920928955078, 0.179118812084198],
          [0.8461772203445435, 0.7952491044998169, 0.6470762491226196, 0.2520409822463989],
          [0.908548891544342, 0.8572604656219482, 0.7078656554222107, 0.3086767792701721]
        ]
      >
  """
  defn stft_to_mel(z, fs, opts \\ []) do
    opts = keyword!(opts, [:nfft, :nmels, :max_mel, :mel_frequency_spacing, type: {:f, 32}])

    magnitudes = Nx.abs(z) ** 2

    filters = mel_filters(opts[:nfft], opts[:nmels], fs, mel_filters_opts(opts))

    freq_size = div(opts[:nfft], 2)

    real_freqs_mag = Nx.slice_along_axis(magnitudes, 0, freq_size, axis: :frequencies)
    real_freqs_filters = Nx.slice_along_axis(filters, 0, freq_size, axis: :frequencies)

    mel_spec =
      Nx.dot(
        real_freqs_mag,
        [:frequencies],
        real_freqs_filters,
        [:frequencies]
      )

    mel_spec = Nx.reshape(mel_spec, Nx.shape(mel_spec), names: [:frames, :mel])

    log_spec = Nx.log(Nx.clip(mel_spec, 1.0e-10, :infinity)) / Nx.log(10)
    log_spec = Nx.max(log_spec, Nx.reduce_max(log_spec) - 8)
    (log_spec + 4) / 4
  end

  deftransformp mel_filters_opts(opts) do
    Keyword.take(opts, [:max_mel, :mel_frequency_spacing, :type])
  end
end
