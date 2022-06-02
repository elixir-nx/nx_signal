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
    * `window` - the window tensor that will be applied to each frame before the DFT.
    * `overlap_size` - the number of samples for the overlap between frames.
    Defaults to `div(frame_size, 2)`.

  ## Examples

      iex> {z, t, f} = NxSignal.stft(Nx.iota({4}), window: NxSignal.Windows.rectangular(n: 2), overlap_size: 1, nfft: 2, fs: 400)
      iex> z
      #Nx.Tensor<
        c64[frequencies: 2][frames: 3]
        [
          [1.0+0.0i, 3.0+0.0i, 5.0+0.0i],
          [-1.0+0.0i, -1.0+0.0i, -1.0+0.0i]
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
  defn stft(data, opts \\ []) do
    {spectrum, frame_size, fs} =
      transform({data, opts}, fn {data, opts} ->
        opts =
          Keyword.validate!(opts, [
            :overlap_size,
            :window,
            fs: 100,
            nfft: :power_of_two
          ])

        fs = opts[:fs] || raise ArgumentError, "missing fs option"

        window = opts[:window] || raise "missing option :window"
        {frame_size} = Nx.shape(window)

        overlap_size = opts[:overlap_size] || div(frame_size, 2)

        frames =
          for frame <- unfold_frames(data, frame_size, overlap_size) do
            frame
            |> Nx.multiply(window)
            |> Nx.fft(length: opts[:nfft])
            |> Nx.new_axis(1)
          end

        spectrum = Nx.concatenate(frames, axis: 1)

        {Nx.reshape(spectrum, spectrum.shape, names: [:frequencies, :frames]), frame_size, fs}
      end)

    {num_frequencies, num_frames} = Nx.shape(spectrum)

    frequencies =
      Nx.iota({num_frequencies}, type: {:f, 64}, names: [:frequencies]) * fs / num_frequencies

    # assign the middle of the equivalent time window as the time for the given frame
    times = (Nx.iota({num_frames}, names: [:frames]) + 1) * frame_size / (2 * fs)

    {spectrum, times, frequencies}
  end

  defp unfold_frames(data, frame_size, overlap_size) do
    {backend, _} = Nx.default_backend()

    stride = frame_size - overlap_size

    if backend == Torchx.Backend do
      data
      |> Torchx.from_nx()
      |> Torchx.unfold(0, frame_size, stride)
      |> Torchx.to_nx()
      |> Nx.to_batched_list(1)
    else
      {len} = Nx.shape(data)

      for start <- 0..(len - frame_size)//stride do
        Nx.slice(data, [start], [frame_size])
      end
    end
  end
end
