defmodule NxSignal.Waveforms do
  @moduledoc """
  Functions that calculate waveforms given a time tensor.
  """
  import Nx.Defn
  import Nx.Constants, only: [pi: 0]

  # square
  # gauss_pulse
  # chirp
  # sweep_poly
  # unit_impulse
  defn sawtooth(t, opts \\ []) do
    opts = keyword!(opts, width: 1)

    width = opts[:width]

    if width < 0 or width > 1 do
      raise ArgumentError, "width must be between 0 and 1, inclusive. Got: #{inspect(width)}"
    end

    tmod = Nx.remainder(t, 2 * pi())
    # in 0..width*2*pi, function is tmod / (pi * w) - 1
    # in width*2*pi..2*pi, function is (pi*(w+1)-tmod) / (pi*(1-w))

    cond do
      width == 1 ->
        tmod / (pi() * width) - 1

      width == 0 ->
        (pi() * (width + 1) - tmod) / (pi() * (1 - width))

      true ->
        Nx.select(
          tmod < 2 * pi() * width,
          tmod / (pi() * width) - 1,
          (pi() * (width + 1) - tmod) / (pi() * (1 - width))
        )
    end
  end
end
