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

    * `:n` - the window length
    * `:type` - the output type. Defaults to `s64`

  ## Examples

      iex> NxSignal.Windows.rectangular(n: 5)
      #Nx.Tensor<
        s64[5]
        [1, 1, 1, 1, 1]
      >

      iex> NxSignal.Windows.rectangular(n: 5, type: :f32)
      #Nx.Tensor<
        f32[5]
        [1.0, 1.0, 1.0, 1.0, 1.0]
      >
  """
  @doc type: :windowing
  defn rectangular(opts \\ []) do
    opts = keyword!(opts, [:n, type: :s64])
    {n, opts} = pop_window_size(opts)
    Nx.broadcast(Nx.tensor(1, type: opts[:type]), {n})
  end

  @doc """
  Bartlett triangular window.

  See also: `triangular/1`

  ## Options

    * `:n` - The window length. Mandatory option.
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.bartlett(n: 3)
      #Nx.Tensor<
        f32[3]
        [0.0, 0.6666666865348816, 0.6666666269302368]
      >
  """
  @doc type: :windowing
  defn bartlett(opts \\ []) do
    opts = keyword!(opts, [:n, :name, type: {:f, 32}])
    {n, opts} = pop_window_size(opts)
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

      iex> NxSignal.Windows.triangular(n: 3)
      #Nx.Tensor<
        f32[3]
        [0.5, 1.0, 0.5]
      >
  """
  @doc type: :windowing
  defn triangular(opts \\ []) do
    opts = keyword!(opts, [:n, :name, type: {:f, 32}])
    {n, opts} = pop_window_size(opts)
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

    * `:n` - The window length. Mandatory option.
    * `:is_periodic` - If `true`, produces a periodic window,
       otherwise produces a symmetric window. Defaults to `true`
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.blackman(n: 5, is_periodic: false)
      #Nx.Tensor<
        f32[5]
        [-1.4901161193847656e-8, 0.3400000333786011, 0.9999999403953552, 0.3400000333786011, -1.4901161193847656e-8]
      >

      iex> NxSignal.Windows.blackman(n: 5, is_periodic: true)
      #Nx.Tensor<
        f32[5]
        [-1.4901161193847656e-8, 0.20077012479305267, 0.8492299318313599, 0.8492299318313599, 0.20077012479305267]
      >

      iex> NxSignal.Windows.blackman(n: 6, is_periodic: true, type: {:f, 32})
      #Nx.Tensor<
        f32[6]
        [-1.4901161193847656e-8, 0.12999999523162842, 0.6299999952316284, 0.9999999403953552, 0.6299999952316284, 0.12999999523162842]
      >
  """
  @doc type: :windowing
  defn blackman(opts \\ []) do
    opts = keyword!(opts, [:n, :name, is_periodic: true, type: {:f, 32}])
    {l, opts} = pop_window_size(opts)
    name = opts[:name]
    type = opts[:type]
    is_periodic = opts[:is_periodic]

    l =
      if is_periodic do
        l + 1
      else
        l
      end

    m = div_ceil(l, 2)

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

    * `:n` - The window length. Mandatory option.
    * `:is_periodic` - If `true`, produces a periodic window,
       otherwise produces a symmetric window. Defaults to `true`
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.hamming(n: 5, is_periodic: true)
      #Nx.Tensor<
        f32[5]
        [0.08000001311302185, 0.39785221219062805, 0.9121478796005249, 0.9121478199958801, 0.3978521227836609]
      >
      iex> NxSignal.Windows.hamming(n: 5, is_periodic: false)
      #Nx.Tensor<
        f32[5]
        [0.08000001311302185, 0.5400000214576721, 1.0, 0.5400000214576721, 0.08000001311302185]
      >
  """
  @doc type: :windowing
  defn hamming(opts \\ []) do
    opts = keyword!(opts, [:n, :name, is_periodic: true, type: {:f, 32}])
    {l, opts} = pop_window_size(opts)
    name = opts[:name]
    type = opts[:type]
    is_periodic = opts[:is_periodic]

    l =
      if is_periodic do
        l + 1
      else
        l
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

    * `:n` - The window length. Mandatory option.
    * `:is_periodic` - If `true`, produces a periodic window,
       otherwise produces a symmetric window. Defaults to `true`
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.hann(n: 5, is_periodic: false)
      #Nx.Tensor<
        f32[5]
        [0.0, 0.5, 1.0, 0.5, 0.0]
      >
      iex> NxSignal.Windows.hann(n: 5, is_periodic: true)
      #Nx.Tensor<
        f32[5]
        [0.0, 0.34549152851104736, 0.9045085310935974, 0.9045084714889526, 0.3454914391040802]
      >
  """
  @doc type: :windowing
  defn hann(opts \\ []) do
    opts = keyword!(opts, [:n, :name, is_periodic: true, type: {:f, 32}])
    {l, opts} = pop_window_size(opts)
    name = opts[:name]
    type = opts[:type]
    is_periodic = opts[:is_periodic]

    l =
      if is_periodic do
        l + 1
      else
        l
      end

    n = Nx.iota({l}, names: [name], type: type)

    window = 0.5 * (1 - Nx.cos(2 * @pi * n / (l - 1)))

    if is_periodic do
      Nx.slice(window, [0], [l - 1])
    else
      window
    end
  end

  deftransformp pop_window_size(opts) do
    {n, opts} = Keyword.pop(opts, :n)

    unless n do
      raise "missing :n option"
    end

    {n, opts}
  end

  deftransformp div_ceil(num, den) do
    ceil(num / den)
  end
end
