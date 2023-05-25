defmodule NxSignal.Waveforms do
  @moduledoc """
  Functions that calculate waveforms given a time tensor.
  """
  import Nx.Defn
  import Nx.Constants, only: [pi: 0]

  @doc ~S"""
  Periodic sawtooth or triangular waveform.

  The wave as a period of $2\pi$, rising from -1 to 1
  in the interval $[0, 2\pi\cdot\text{width}]$ and dropping from
  1 to -1 in the interval $[2\pi \cdot \text{width}, 2\pi]$.

  ## Options

    * `:width` - the width of the sawtooth. Must be a number
    between 0 and 1 (both inclusive). Defaults to 1.

  ## Examples

  A 5Hz waveform sampled at 500Hz for 1 second can be defined as:

      t = Nx.linspace(0, 1, n: 500)
      n = Nx.multiply(2 * :math.pi() * 5, t)
      wave = NxSignal.Waveforms.sawtooth(n)
  """
  @doc type: :waveforms
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
      iex> duty = Nx.tensor([0.1, 0, 0.3, 0, 0.5, 0, 0.7, 0, 0.9, 0])
      iex> NxSignal.Waveforms.square(t, duty: duty)
      #Nx.Tensor<
        s64[10]
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
      >
  """
  @doc type: :waveforms
  deftransform square(t, opts \\ []) do
    opts = Keyword.validate!(opts, duty: 0.5)
    square_n(t, opts[:duty])
  end

  defnp square_n(t, duty) do
    tmod = Nx.remainder(t, 2 * pi())
    Nx.select(tmod < duty * 2 * pi(), 1, -1)
  end

  @doc ~S"""
  Gaussian modulated sinusoid.

  The returned value follows the formula:

  $$
  f(t) = e^{-at^2}(cos(2 \pi f_c t) + isin(2 \pi f_c t))
  $$

  Where the exponential envelope is returned as `envelope`,
  and the real and imaginary parts of $f(t)$ are returned as
  `in_phase` and `quadrature` in the output map.

  Note that `in_phase` and `quadrature` are are equivalent to
  $\operatorname{Re} \lbrace f(t) \rbrace$ and $\operatorname{Im} \lbrace f(t) \rbrace$ respectively.

  ## Examples

      iex> t = Nx.linspace(0, 1, n: 4)
      iex> pulse = NxSignal.Waveforms.gaussian_pulse(t, center_frequency: 4)
      iex> pulse.envelope
      #Nx.Tensor<
        f32[4]
        [1.0, 0.20443114638328552, 0.001746579073369503, 6.236254534996988e-7]
      >
      iex> pulse.in_phase
      #Nx.Tensor<
        f32[4]
        [1.0, -0.10221561044454575, -8.732887799851596e-4, 6.236254534996988e-7]
      >
      iex> pulse.quadrature
      #Nx.Tensor<
        f32[4]
        [0.0, 0.17704254388809204, -0.0015125821810215712, 4.361525517918713e-13]
      >

      iex> t = Nx.linspace(0, 1, n: 4)
      iex> pulse = NxSignal.Waveforms.gaussian_pulse(t, center_frequency: 4, bandwidth: 0.25)
      iex> pulse.envelope
      #Nx.Tensor<
        f32[4]
        [1.0, 0.6724140048027039, 0.20443114638328552, 0.028101593255996704]
      >
      iex> pulse.in_phase
      #Nx.Tensor<
        f32[4]
        [1.0, -0.3362071216106415, -0.1022154912352562, 0.028101593255996704]
      >
      iex> pulse.quadrature
      #Nx.Tensor<
        f32[4]
        [0.0, 0.5823275446891785, -0.177042618393898, 1.9653754179671523e-8]
      >
  """
  @doc type: :waveforms
  defn gaussian_pulse(t, opts \\ []) do
    opts =
      keyword!(opts,
        center_frequency: 1000,
        bandwidth: 0.5,
        bandwidth_reference_level: -6
      )

    fc = opts[:center_frequency]
    bw = opts[:bandwidth]
    bwr = opts[:bandwidth_reference_level]

    if fc < 0 do
      raise ArgumentError,
            "Center frequency must be greater than or equal to 0, got: #{inspect(fc)}"
    end

    if bw <= 0 do
      raise ArgumentError,
            "Bandwidth must be greater than 0, got: #{inspect(bw)}"
    end

    if bwr >= 0 do
      raise ArgumentError,
            "Bandwidth reference level must be less than 0, got: #{inspect(bwr)}"
    end

    ref = 10 ** (bwr / 20)

    a = -((pi() * fc * bw) ** 2) / (4.0 * Nx.log(ref))

    yenv = Nx.exp(-a * t * t)
    yarg = 2 * pi() * fc * t
    yI = yenv * Nx.cos(yarg)
    yQ = yenv * Nx.sin(yarg)

    %{envelope: yenv, in_phase: yI, quadrature: yQ}
  end

  @doc """
  Chirp function.

  Starts at `t` with frequency `f0` and ends at `t1` with
  frequency `f1`.

  ## Options

    * `:phi` - phase shift for the chirp.
    * `:vertex_zero` - determines the position of the parabolic vertex
      for when `method: :quadratic`. Defaults to `true`.
    * `:method` - One of various frequency interpolation methods:
      * `:linear` - linear interpolation.
      * `:quadratic` - parabolic interpolation with vertex at `t1` or `t0`,
        depending if `vertex_zero: false` or `vertex_zero: true` respectively.
      * `:hyperbolic` - hyperbolic interpolation.
      * `:logarithmic` - logarithmic (also known as geometric or exponential)
        interpolation. `f0` and `f1` must be non-zero and have the same sign.

  ## Examples

      iex> t = Nx.linspace(0, 10, n: 5)
      iex> NxSignal.Waveforms.chirp(t, 10, 10, 1, method: :linear)
      #Nx.Tensor<
        f32[5]
        [1.0, 0.38268470764160156, 3.795033308051643e-6, -0.382683128118515, 1.0]
      >
      iex> NxSignal.Waveforms.chirp(t, 10, 10, 1, method: :quadratic)
      #Nx.Tensor<
        f32[5]
        [1.0, -0.9807833433151245, -9.958475288840418e-8, -0.5555803775787354, 1.0]
      >
      iex> NxSignal.Waveforms.chirp(t, 10, 10, 1, method: :quadratic, vertex_zero: false)
      #Nx.Tensor<
        f32[5]
        [1.0, 0.5555850863456726, -7.490481493732659e-6, 0.98078453540802, 1.0]
      >
      iex> NxSignal.Waveforms.chirp(t, 10, 10, 1, method: :hyperbolic)
      #Nx.Tensor<
        f32[5]
        [1.0, 0.8229323029518127, 0.9335360527038574, 0.013466471806168556, -0.8630329966545105]
      >
      iex> NxSignal.Waveforms.chirp(t, 10, 10, 1, method: :logarithmic)
      #Nx.Tensor<
        f32[5]
        [1.0, 0.9989554286003113, -0.33371755480766296, -0.2700612545013428, 0.8558982610702515]
      >
  """
  @doc type: :waveforms
  defn chirp(t, f0, t1, f1, opts \\ []) do
    opts = keyword!(opts, phi: 0, vertex_zero: true, method: :linear)

    phase =
      case {chirp_validate_method(opts[:method]), opts[:vertex_zero]} do
        {:linear, _} ->
          beta = (f1 - f0) / t1
          2 * pi() * (f0 * t + 0.5 * beta * t ** 2)

        {:quadratic, true} ->
          beta = (f1 - f0) / t1 ** 2
          2 * pi() * (f0 * t + beta * t ** 3 / 3)

        {:quadratic, _} ->
          beta = (f1 - f0) / t1 ** 2
          2 * pi() * (f1 * t + beta * ((t1 - t) ** 3 - t1 ** 3) / 3)

        {:logarithmic, _} ->
          cond do
            f0 * f1 <= 0 ->
              Nx.broadcast(:nan, t.shape)

            f0 == f1 ->
              2 * pi() * f0 * t

            true ->
              beta = t1 / Nx.log(f1 / f0)
              2 * pi() * beta * f0 * ((f1 / f0) ** (t / t1) - 1.0)
          end

        {:hyperbolic, _} ->
          if f0 == f1 do
            2 * pi() * f0 * t
          else
            singular_point = -f1 * t1 / (f0 - f1)
            2 * pi() * (-singular_point * f0) * Nx.log(Nx.abs(1 - t / singular_point))
          end
      end

    Nx.cos(phase + opts[:phi])
  end

  deftransformp chirp_validate_method(method) do
    valid_methods = [:linear, :quadratic, :logarithmic, :hyperbolic]

    unless method in valid_methods do
      raise ArgumentError,
            "invalid method, must be one of #{inspect(valid_methods)}, got: #{inspect(method)}"
    end

    method
  end

  @doc """
  Frequency-swept cosine generator, with a time-dependent frequency.

  This function generates a sinusoidal function whose instantaneous
  frequency varies with time. The frequency at time `t` is given by
  the polynomial specified by the coefficients contained in `coefs`.

  See also: `chirp/5`

  ## Options

    * `:phi` - phase shift to be applied before calculating the `Nx.cos`
      for the output. Defaults to 0.
    * `:phi_unit` - determines if `:phi` is given in `:radians` or `:degrees`.
      Defaults to `:radians`.

  ## Examples

      iex> t = Nx.linspace(0, 10, n: 5)
      iex> NxSignal.Waveforms.polynomial_sweep(t, Nx.tensor([2, 0, 1]))
      #Nx.Tensor<
        f32[5]
        [1.0, 0.866027295589447, -0.500006377696991, 1.7942518752533942e-5, -0.49998921155929565]
      >
      iex> NxSignal.Waveforms.polynomial_sweep(t, Nx.tensor([2, 0, 1]), phi: :math.pi() / 2)
      #Nx.Tensor<
        f32[5]
        [-4.371138828673793e-8, 0.499999463558197, -0.8660194873809814, 1.0, 0.8660338521003723]
      >
      iex> NxSignal.Waveforms.polynomial_sweep(t, Nx.tensor([1, 0]))
      #Nx.Tensor<
        f32[5]
        [1.0, 0.7071065306663513, -1.0, 0.7071084976196289, 1.0]
      >
      iex> NxSignal.Waveforms.polynomial_sweep(t, Nx.tensor([1, 0]), phi: 180, phi_unit: :degrees)
      #Nx.Tensor<
        f32[5]
        [-1.0, -0.7071069478988647, 1.0, -0.7071129679679871, -1.0]
      >
  """
  @doc type: :waveforms
  defn polynomial_sweep(t, coefs, opts \\ []) do
    opts = keyword!(opts, phi: 0, phi_unit: :radians)
    {n} = Nx.shape(coefs)
    # assumes t is of shape {m}
    iota = n - Nx.iota({n})
    t_poly = t ** Nx.new_axis(iota, 1)

    int_coefs = coefs / iota

    phase = Nx.dot(int_coefs, t_poly)

    phi =
      case {opts[:phi], opts[:phi_unit]} do
        {phi, :radians} -> phi
        {phi, :degrees} -> phi * pi() / 180
      end

    Nx.cos(2 * pi() * phase + phi)
  end

  @doc """
  Discrete delta function or unit basis vector.

  ## Options

    * `:index` - one of number, numerical tensor
      with length equal to the rank of the given
      shape, or `:midpoint`. `index: :midpoint`,
      is a shortcut for inserting the impulse
      at the index which corresponds to half of
      each dimension. Defaults to 0.

    * `:type` - datatype for the output. Defaults to `:f32`.

  ## Examples

      iex> NxSignal.Waveforms.unit_impulse({2})
      #Nx.Tensor<
        f32[2]
        [1.0, 0.0]
      >

      iex> NxSignal.Waveforms.unit_impulse({3, 5}, type: :s64, index: :midpoint)
      #Nx.Tensor<
        s64[3][5]
        [
          [0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0]
        ]
      >

      iex> NxSignal.Waveforms.unit_impulse({3, 5}, index: Nx.tensor([[2, 3]]), type: :s64)
      #Nx.Tensor<
        s64[3][5]
        [
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0]
        ]
      >
  """
  @doc type: :waveforms
  deftransform unit_impulse(shape, opts \\ []) do
    opts = Keyword.validate!(opts, index: 0, type: :f32)
    index = unit_impulse_index(shape, opts[:index])

    unit_impulse_n(index, Keyword.put(opts, :shape, shape))
  end

  defnp unit_impulse_n(index, opts \\ []) do
    shape = opts[:shape]
    type = opts[:type]

    zero = Nx.tensor(0, type: type)

    zeros = Nx.broadcast(zero, shape)

    Nx.indexed_put(zeros, index, 1)
  end

  deftransformp unit_impulse_index(shape, index) do
    n = Nx.rank(shape)

    case index do
      :midpoint ->
        shape
        |> Tuple.to_list()
        |> Enum.map(&div(&1, 2))
        |> then(&Nx.tensor(&1))

      index ->
        Nx.reshape(index, {n})
    end
  end
end
