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
  @doc """
  Periodic sawtooth or triangular waveform.

  The wave as a period of $2\\pi$, rising from -1 to 1
  on the interval $[0, 2\\piwidth]$ and dropping from
  1 to -1 on the interval $[2\\piwidth, 2\\pi]$.

  ## Options

    * `:width` - the width of the sawtooth. Must be a number
    between 0 and 1 (both inclusive). Defaults to 1.

  ## Examples

  A 5Hz waveform sampled at 500Hz for 1 second can be defined as:

      t = Nx.linspace(0, 1, n: 500)
      n = Nx.multiply(2 * :math.pi() * 5, t)
      wave = NxSignal.Waveforms.sawtooth(n)
  """
  defn sawtooth(t, opts \\ []) do
    opts = keyword!(opts, width: 1)

    width = opts[:width]

    if width < 0 or width > 1 do
      raise ArgumentError, "width must be between 0 and 1, inclusive. Got: #{inspect(width)}"
    end

    tmod = Nx.remainder(t, 2 * pi())

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
