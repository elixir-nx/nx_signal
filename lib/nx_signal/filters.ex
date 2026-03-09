defmodule NxSignal.Filters do
  @moduledoc """
  Common filter functions.
  """
  import Nx.Defn
  import NxSignal.Convolution

  @doc ~S"""
  Performs a median filter on a tensor.

  ## Options

    * `:kernel_shape` - the shape of the sliding window.
    It must be compatible with the shape of the tensor.
  """
  @doc type: :filters
  defn median(t, opts) do
    validate_median_opts!(t, opts)

    idx =
      t
      |> idx_tensor()
      |> Nx.vectorize(:elements)

    t
    |> Nx.slice(start_indices(t, idx), kernel_lengths(opts[:kernel_shape]))
    |> Nx.median()
    |> Nx.devectorize(keep_names: false)
    |> Nx.reshape(t.shape)
    |> Nx.as_type({:f, 32})
  end

  deftransformp validate_median_opts!(t, opts) do
    Keyword.validate!(opts, [:kernel_shape])

    if Nx.rank(t) != Nx.rank(opts[:kernel_shape]) do
      raise ArgumentError, message: "kernel shape must be of the same rank as the tensor"
    end
  end

  deftransformp idx_tensor(t) do
    t
    |> Nx.axes()
    |> Enum.map(&Nx.iota(t.shape, axis: &1))
    |> Nx.stack(axis: -1)
    |> Nx.reshape({:auto, length(Nx.axes(t))})
  end

  deftransformp start_indices(t, idx_tensor) do
    t
    |> Nx.axes()
    |> Enum.map(&idx_tensor[&1])
  end

  deftransformp kernel_lengths(kernel_shape), do: Tuple.to_list(kernel_shape)

  @doc """
  Applies a Wiener filter to the given Nx tensor.

  ## Options

      * `:kernel_size` - filter size given either a number or a tuple. 
        If a number is given, a kernel with the given size, and same number of axes 
        as the input tensor will be used. Defaults to `3`.
      * `:noise` - noise power, given as a scalar. This will be estimated based on the input tensor if `nil`. Defaults to `nil`.

  ## Examples

      iex> t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
      iex> NxSignal.Filters.wiener(t, kernel_size: {2, 2}, noise: 10)
      #Nx.Tensor<
        f32[3][3]
        [
          [0.25, 0.75, 1.25],
          [1.25, 3.0, 4.0],
          [2.75, 6.0, 7.0]
        ]
      >
  """
  @doc type: :filters
  deftransform wiener(t, opts \\ []) do
    # Validate and extract options
    opts = Keyword.validate!(opts, noise: nil, kernel_size: 3)

    rank = Nx.rank(t)
    kernel_size = Keyword.fetch!(opts, :kernel_size)
    noise = Keyword.fetch!(opts, :noise)

    # Ensure `kernel_size` is a tuple
    kernel_size =
      cond do
        is_integer(kernel_size) -> Tuple.duplicate(kernel_size, rank)
        is_tuple(kernel_size) -> kernel_size
        true -> raise ArgumentError, "kernel_size must be an integer or tuple"
      end

    # Convert `nil` noise to `0.0` so it's always a valid tensor
    noise_t = if is_nil(noise), do: Nx.tensor(0.0), else: Nx.tensor(noise)

    # Compute filter window size
    size = Tuple.to_list(kernel_size) |> Enum.reduce(1, &*/2)

    # Ensure the kernel is the same size as the filter window
    kernel = Nx.broadcast(1.0, kernel_size)

    t
    |> Nx.as_type(:f64)
    |> wiener_n(kernel, noise_t, calculate_noise: is_nil(noise), size: size)
    |> Nx.as_type(Nx.type(t))
  end

  @doc """
  FIR filter design using the window method.

  Computes the coefficients of a finite impulse response (FIR) filter.
  The filter has linear phase (Type I for odd `num_taps`, Type II for even).
  Type II filters have a zero at Nyquist, so a filter requiring gain there
  must use an odd number of taps.

  ## Arguments

    * `num_taps` - number of filter coefficients (filter order + 1).
    * `cutoff` - list of cutoff frequencies in the same units as `sampling_rate`.

  ## Options

    * `:window` - window function to apply. One of `:hamming`, `:hann`,
      `:blackman`, `:bartlett`, `:rectangular`, or `{:kaiser, beta}`.
      Defaults to `:hamming`.
    * `:pass_zero` - if `true`, the DC gain is 1 (lowpass or bandstop);
      if `false`, the DC gain is 0 (highpass or bandpass). Defaults to `true`.
    * `:scale` - if `true`, normalise the coefficients so the frequency
      response is exactly 1 at a reference frequency. Defaults to `true`.
    * `:sampling_rate` - the sampling rate in Hz; `cutoff` is given in the
      same units. Defaults to `2.0` so cutoffs are already normalised to
      `[0, 1]` where `1` equals Nyquist.
    * `:type` - output tensor type. Defaults to `{:f, 32}`.

  ## Examples

      iex> coeffs = NxSignal.Filters.firwin(5, [0.3], window: :hamming, sampling_rate: 2.0)
      iex> Nx.shape(coeffs)
      {5}

  """
  @doc type: :filters
  deftransform firwin(num_taps, cutoff, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        window: :hamming,
        pass_zero: true,
        scale: true,
        sampling_rate: 2.0,
        type: {:f, 32}
      )

    type = opts[:type]
    nyq = opts[:sampling_rate] / 2.0

    if not is_list(cutoff) do
      raise ArgumentError, "cutoff must be a list of frequencies, got: #{inspect(cutoff)}"
    end

    cutoff_list = cutoff |> Enum.map(&(&1 / nyq)) |> Enum.sort()

    # Validate cutoffs are strictly in (0, 1) — after sorting, only the extremes need checking
    first_cutoff = List.first(cutoff_list)
    last_cutoff = List.last(cutoff_list)

    if first_cutoff <= 0.0 do
      raise ArgumentError,
            "cutoff must be strictly between 0 and Nyquist (exclusive), got: #{first_cutoff * nyq}"
    end

    if last_cutoff >= 1.0 do
      raise ArgumentError,
            "cutoff must be strictly between 0 and Nyquist (exclusive), got: #{last_cutoff * nyq}"
    end

    # Type II filters (even num_taps) have a zero at Nyquist.
    # Any filter requiring gain there must use an odd number of taps.
    n_cuts = length(cutoff_list)
    even_n_cuts = rem(n_cuts, 2) == 0

    nyquist_gain =
      (opts[:pass_zero] and even_n_cuts) or
        (not opts[:pass_zero] and not even_n_cuts)

    if nyquist_gain and rem(num_taps, 2) == 0 do
      raise ArgumentError,
            "a filter with non-zero gain at Nyquist (e.g. highpass) requires " <>
              "an odd number of taps, got: #{num_taps}"
    end

    m = (num_taps - 1) / 2.0
    alpha = Nx.subtract(Nx.iota({num_taps}, type: type), m)

    # Build ideal impulse response by summing contributions from each passband.
    # Passband pairs are selected at compile time from [0, c1, c2, ..., cn, 1].
    all_freqs = [0.0 | cutoff_list] ++ [1.0]

    h =
      all_freqs
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index()
      |> Enum.filter(fn {_pair, i} ->
        if opts[:pass_zero], do: rem(i, 2) == 0, else: rem(i, 2) == 1
      end)
      |> Enum.reduce(Nx.broadcast(Nx.tensor(0.0, type: type), {num_taps}), fn {[a, b], _i}, acc ->
        firwin_contribution(a, b, alpha, acc)
      end)

    w = firwin_build_window(num_taps, opts[:window], type)
    h = Nx.multiply(h, w)

    if opts[:scale] do
      firwin_scale(h, alpha, cutoff_list, opts[:pass_zero])
    else
      h
    end
  end

  defnp firwin_contribution(a, b, alpha, acc) do
    contribution_a = a * NxSignal.Waveforms.sinc(a * alpha)
    contribution_b = b * NxSignal.Waveforms.sinc(b * alpha)
    acc + contribution_b - contribution_a
  end

  deftransformp firwin_scale(h, alpha, cutoff_list, pass_zero) do
    scale_freq =
      cond do
        pass_zero ->
          0.0

        match?([_], cutoff_list) ->
          1.0

        true ->
          [first, second | _] = cutoff_list
          (first + second) / 2.0
      end

    scale_factor =
      Nx.abs(
        Nx.dot(
          h,
          Nx.cos(Nx.multiply(alpha, :math.pi() * scale_freq))
        )
      )

    Nx.divide(h, scale_factor)
  end

  deftransformp firwin_build_window(num_taps, window, type) do
    case window do
      :hamming ->
        NxSignal.Windows.hamming(num_taps, is_periodic: false, type: type)

      :hann ->
        NxSignal.Windows.hann(num_taps, is_periodic: false, type: type)

      :blackman ->
        NxSignal.Windows.blackman(num_taps, is_periodic: false, type: type)

      :bartlett ->
        NxSignal.Windows.bartlett(num_taps, type: type)

      :rectangular ->
        NxSignal.Windows.rectangular(num_taps, type: type)

      {:kaiser, beta} ->
        NxSignal.Windows.kaiser(num_taps, beta: beta, is_periodic: false, type: type)

      _ ->
        raise ArgumentError,
              "unknown window #{inspect(window)}, supported: " <>
                ":hamming, :hann, :blackman, :bartlett, :rectangular, {:kaiser, beta}"
    end
  end

  defnp wiener_n(t, kernel, noise, opts) do
    size = opts[:size]

    # Compute local mean using "same" mode in correlation
    l_mean = correlate(t, kernel, mode: :same) / size

    # Compute local variance
    l_var =
      correlate(t ** 2, kernel, mode: :same)
      |> Nx.divide(size)
      |> Nx.subtract(l_mean ** 2)

    # Ensure `noise` is a tensor to avoid `nil` issues in `defnp`
    noise =
      case opts[:calculate_noise] do
        true -> Nx.mean(l_var)
        false -> noise
      end

    # Apply Wiener filter formula
    res = (t - l_mean) * (1 - noise / l_var)
    Nx.select(l_var < noise, l_mean, res + l_mean)
  end
end
