defmodule NxSignal.Waveforms do
  @moduledoc """
  Functions that calculate waveforms given a time tensor.
  """
  import Nx.Defn
  import Nx.Constants, only: [pi: 0]

  # gauss_pulse
  # chirp
  # sweep_poly
  # unit_impulse
  @doc """
  Periodic sawtooth or triangular waveform.

  The wave as a period of $2\\pi$, rising from -1 to 1
  in the interval $[0, 2\\piwidth]$ and dropping from
  1 to -1 in the interval $[2\\piwidth, 2\\pi]$.

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

  @doc """
  A periodic square wave with period $2\\pi$.

  Evaluates to 1 in the interval $[0, 2\\pi\\text{duty}]$
  and -1 in the interval $[2\\pi\\text{duty}, 2\\pi]$.

  ## Options

    * `:duty` - a number or tensor representing the duty cycle.
    If a tensor is given, the waveform changes over time, and it
    must have the same length as the `t` input. Defaults to `0.5`.

  ## Examples

      iex> t = Nx.iota({10}) |> Nx.multiply(:math.pi() * 2 / 10)
      iex> NxSignal.Waveforms.square(t, duty: 0.1)
      #Nx.Tensor<
        s64[10]
        [1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
      >
      iex> NxSignal.Waveforms.square(t, duty: 0.5)
      #Nx.Tensor<
        s64[10]
        [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
      >
      iex> NxSignal.Waveforms.square(t, duty: 1)
      #Nx.Tensor<
        s64[10]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      >

      iex> t = Nx.iota({10}) |> Nx.multiply(:math.pi() * 2 / 10)
      iex> Nx.tensor([0.1, 0, 0.3, 0, 0.5, 0, 0.7, 0, 0.9, 0])
      iex> NxSignal.Waveforms.square(t, duty: duty)
      #Nx.Tensor<
        s64[10]
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
      >
  """
  deftransform square(t, opts \\ []) do
    opts = Keyword.validate!(opts, duty: 0.5)
    square_n(t, opts[:duty])
  end

  defnp square_n(t, duty) do
    tmod = Nx.remainder(t, 2 * pi())
    Nx.select(tmod < duty * 2 * pi(), 1, -1)
  end
end
