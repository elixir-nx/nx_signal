defmodule NxSignal.Windows do
  @moduledoc """
  Common window functions.
  """
  import Nx.Defn

  @pi :math.pi()

  @doc """
  Rectangular window.

  Useful for when no window function should be applied.

  ## Options

    * `:type` - the output type. Defaults to `s64`

  ## Examples

      iex> NxSignal.Windows.rectangular(5)
      #Nx.Tensor<
        s64[5]
        [1, 1, 1, 1, 1]
      >

      iex> NxSignal.Windows.rectangular(5, type: :f32)
      #Nx.Tensor<
        f32[5]
        [1.0, 1.0, 1.0, 1.0, 1.0]
      >
  """
  @doc type: :windowing
  deftransform rectangular(n, opts \\ []) when is_integer(n) do
    opts = Keyword.validate!(opts, type: :s64)
    Nx.broadcast(Nx.tensor(1, type: opts[:type]), {n})
  end

  @doc """
  Bartlett triangular window.

  See also: `triangular/1`

  ## Options

    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.bartlett(3)
      #Nx.Tensor<
        f32[3]
        [0.0, 0.6666666865348816, 0.6666666269302368]
      >
  """
  @doc type: :windowing
  deftransform bartlett(n, opts \\ []) when is_integer(n) do
    opts = Keyword.validate!(opts, type: {:f, 32})
    bartlett_n(Keyword.put(opts, :n, n))
  end

  defnp bartlett_n(opts) do
    n = opts[:n]
    name = opts[:name]
    type = opts[:type]

    n_on_2 = div(n, 2)
    left_size = n_on_2 + rem(n, 2)
    left_idx = Nx.iota({left_size}, names: [name], type: type)
    right_idx = Nx.iota({n_on_2}, names: [name], type: type) + left_size

    Nx.concatenate([
      left_idx * 2 / n,
      2 - right_idx * 2 / n
    ])
  end

  @doc """
  Triangular window.

  See also: `bartlett/1`

  ## Options

    * `:n` - The window length. Mandatory option.
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.triangular(3)
      #Nx.Tensor<
        f32[3]
        [0.5, 1.0, 0.5]
      >
  """
  @doc type: :windowing
  deftransform triangular(n, opts \\ []) when is_integer(n) do
    opts = Keyword.validate!(opts, [:name, type: {:f, 32}])
    triangular_n(Keyword.put(opts, :n, n))
  end

  defnp triangular_n(opts) do
    n = opts[:n]
    name = opts[:name]
    type = opts[:type]

    case rem(n, 2) do
      1 ->
        # odd case
        n_on_2 = div(n + 1, 2)

        idx = Nx.iota({n_on_2}, names: [name], type: type) + 1

        left = idx * 2 / (n + 1)
        Nx.concatenate([left, left |> Nx.reverse() |> Nx.slice([1], [Nx.size(left) - 1])])

      0 ->
        # even case
        n_on_2 = div(n + 1, 2)

        idx = Nx.iota({n_on_2}, names: [name], type: type) + 1

        left = (2 * idx - 1) / n
        Nx.concatenate([left, Nx.reverse(left)])
    end
  end

  @doc """
  Blackman window.

  ## Options

    * `:is_periodic` - If `true`, produces a periodic window,
       otherwise produces a symmetric window. Defaults to `true`
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.blackman(5, is_periodic: false)
      #Nx.Tensor<
        f32[5]
        [-1.4901161193847656e-8, 0.3400000333786011, 0.9999999403953552, 0.3400000333786011, -1.4901161193847656e-8]
      >

      iex> NxSignal.Windows.blackman(5, is_periodic: true)
      #Nx.Tensor<
        f32[5]
        [-1.4901161193847656e-8, 0.20077012479305267, 0.8492299318313599, 0.8492299318313599, 0.20077012479305267]
      >

      iex> NxSignal.Windows.blackman(6, is_periodic: true, type: {:f, 32})
      #Nx.Tensor<
        f32[6]
        [-1.4901161193847656e-8, 0.12999999523162842, 0.6299999952316284, 0.9999999403953552, 0.6299999952316284, 0.12999999523162842]
      >
  """
  @doc type: :windowing
  deftransform blackman(n, opts \\ []) when is_integer(n) do
    opts = Keyword.validate!(opts, [:name, is_periodic: true, type: {:f, 32}])
    blackman_n(Keyword.put(opts, :n, n))
  end

  defnp blackman_n(opts) do
    n = opts[:n]
    name = opts[:name]
    type = opts[:type]
    is_periodic = opts[:is_periodic]

    l =
      if is_periodic do
        n + 1
      else
        n
      end

    m =
      integer_div_ceil(l, 2)

    n = Nx.iota({m}, names: [name], type: type)

    left =
      0.42 - 0.5 * Nx.cos(2 * @pi * n / (l - 1)) +
        0.08 * Nx.cos(4 * @pi * n / (l - 1))

    window =
      if rem(l, 2) == 0 do
        Nx.concatenate([left, Nx.reverse(left)])
      else
        Nx.concatenate([left, left |> Nx.reverse() |> Nx.slice([1], [Nx.size(left) - 1])])
      end

    if is_periodic do
      Nx.slice(window, [0], [Nx.size(window) - 1])
    else
      window
    end
  end

  @doc """
  Hamming window.

  ## Options

    * `:is_periodic` - If `true`, produces a periodic window,
       otherwise produces a symmetric window. Defaults to `true`
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.hamming(5, is_periodic: true)
      #Nx.Tensor<
        f32[5]
        [0.08000001311302185, 0.39785221219062805, 0.9121478796005249, 0.9121478199958801, 0.3978521227836609]
      >
      iex> NxSignal.Windows.hamming(5, is_periodic: false)
      #Nx.Tensor<
        f32[5]
        [0.08000001311302185, 0.5400000214576721, 1.0, 0.5400000214576721, 0.08000001311302185]
      >
  """
  @doc type: :windowing
  deftransform hamming(n, opts \\ []) when is_integer(n) do
    opts = Keyword.validate!(opts, [:name, is_periodic: true, type: {:f, 32}])
    hamming_n(Keyword.put(opts, :n, n))
  end

  defnp hamming_n(opts) do
    n = opts[:n]
    name = opts[:name]
    type = opts[:type]
    is_periodic = opts[:is_periodic]

    l =
      if is_periodic do
        n + 1
      else
        n
      end

    n = Nx.iota({l}, names: [name], type: type)

    window = 0.54 - 0.46 * Nx.cos(2 * @pi * n / (l - 1))

    if is_periodic do
      Nx.slice(window, [0], [l - 1])
    else
      window
    end
  end

  @doc """
  Hann window.

  ## Options

    * `:is_periodic` - If `true`, produces a periodic window,
       otherwise produces a symmetric window. Defaults to `true`
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.hann(5, is_periodic: false)
      #Nx.Tensor<
        f32[5]
        [0.0, 0.5, 1.0, 0.5, 0.0]
      >
      iex> NxSignal.Windows.hann(5, is_periodic: true)
      #Nx.Tensor<
        f32[5]
        [0.0, 0.34549152851104736, 0.9045085310935974, 0.9045084714889526, 0.3454914391040802]
      >
  """
  @doc type: :windowing
  deftransform hann(n, opts \\ []) when is_integer(n) do
    opts = Keyword.validate!(opts, [:name, is_periodic: true, type: {:f, 32}])
    hann_n(Keyword.put(opts, :n, n))
  end

  defnp hann_n(opts) do
    n = opts[:n]
    name = opts[:name]
    type = opts[:type]
    is_periodic = opts[:is_periodic]

    l =
      if is_periodic do
        n + 1
      else
        n
      end

    n = Nx.iota({l}, names: [name], type: type)

    window = 0.5 * (1 - Nx.cos(2 * @pi * n / (l - 1)))

    if is_periodic do
      Nx.slice(window, [0], [l - 1])
    else
      window
    end
  end

  @doc """
  Creates a Kaiser window of size `window_length`.

  The Kaiser window is a taper formed by using a Bessel function.

  ## Options

    * `:is_periodic` - If `true`, produces a periodic window,
       otherwise produces a symmetric window. Defaults to `true`
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:beta` - Shape parameter for the window. As beta increases, the window becomes more focused in frequency domain. Defaults to 12.0.
    * `:eps` - Epsilon value to avoid division by zero. Defaults to 1.0e-7.
    * `:axis_name` - the axis name. Defaults to `nil`

  ## Examples
      iex> NxSignal.Windows.kaiser(4, beta: 12.0, is_periodic: true)
      #Nx.Tensor<
        f32[4]
        [5.2776191296288744e-5, 0.21566666662693024, 1.0, 0.21566666662693024]
      >

      iex> NxSignal.Windows.kaiser(5, beta: 12.0, is_periodic: true)
      #Nx.Tensor<
        f32[5]
        [5.2776191296288744e-5, 0.10171464085578918, 0.7929369807243347, 0.7929369807243347, 0.10171464085578918]
      >

      iex> NxSignal.Windows.kaiser(4, beta: 12.0, is_periodic: false)
      #Nx.Tensor<
        f32[4]
        [5.2776191296288744e-5, 0.5188394784927368, 0.5188390612602234, 5.2776191296288744e-5]
      >
  """
  @doc type: :windowing
  deftransform kaiser(n, opts \\ []) when is_integer(n) do
    opts =
      Keyword.validate!(opts, [:name, eps: 1.0e-7, beta: 12.0, is_periodic: true, type: {:f, 32}])

    kaiser_n(Keyword.put(opts, :n, n))
  end

  defnp kaiser_n(opts) do
    n = opts[:n]
    name = opts[:name]
    type = opts[:type]
    beta = opts[:beta]
    eps = opts[:eps]
    is_periodic = opts[:is_periodic]

    window_length = if is_periodic, do: n + 1, else: n

    ratio = Nx.linspace(-1, 1, n: window_length, endpoint: true, type: type, name: name)
    sqrt_arg = Nx.max(1 - ratio ** 2, eps)
    r = beta * Nx.sqrt(sqrt_arg)

    window = kaiser_bessel_i0(r) / kaiser_bessel_i0(beta)

    if is_periodic do
      Nx.slice(window, [0], [n])
    else
      window
    end
  end

  defnp kaiser_bessel_i0(x) do
    abs_x = Nx.abs(x)

    small_x_result =
      1 +
        abs_x ** 2 / 4 +
        abs_x ** 4 / 64 +
        abs_x ** 6 / 2304 +
        abs_x ** 8 / 147_456

    large_x_result =
      Nx.exp(abs_x) / Nx.sqrt(2 * Nx.Constants.pi() * abs_x) *
        (1 + 1 / (8 * abs_x) + 9 / (128 * Nx.pow(abs_x, 2)))

    Nx.select(abs_x < 3.75, small_x_result, large_x_result)
  end

  deftransformp integer_div_ceil(num, den) do
    if is_integer(num) and is_integer(den) do
      rem = rem(num, den)

      if rem == 0 do
        div(num, den)
      else
        div(num, den) + 1
      end
    else
      ceil(num / den)
    end
  end
end
