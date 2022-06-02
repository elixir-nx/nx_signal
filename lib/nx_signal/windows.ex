defmodule NxSignal.Windows do
  @moduledoc """
  Definitions for common window functions
  """
  import Nx.Defn

  @doc """
  Rectangular window

  Useful for when no window function should be applied.

  Expects `:n`, the window length, to be passed as an option.
  Also accepts the same options as `Nx.broadcast`.
  """
  defn rectangular(opts \\ []) do
    {n, opts} =
      transform(opts, fn opts ->
        {n, opts} = Keyword.pop(opts, :n)

        unless n do
          raise "missing :n option"
        end

        {n, opts}
      end)

    Nx.broadcast(1, {n}, opts)
  end

  @doc """
  Bartlett triangular window

  See also: `triangular/1`

  ## Options

    * `:n` - The window length. Mandatory option.
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.bartlett(n: 3)
      #Nx.Tensor<
        f64[3]
        [0.0, 0.6666666865348816, 0.6666666269302368]
      >
  """
  defn bartlett(opts \\ []) do
    transform(opts, fn opts ->
      opts = Keyword.validate!(opts, [:n, :name, type: {:f, 64}])

      n = opts[:n]
      name = opts[:name]
      type = opts[:type]

      unless n do
        raise "missing :n option"
      end

      bartlett(n, name, type)
    end)
  end

  defp bartlett(n, name, type) do
    n_on_2 = div(n, 2)
    left_size = n_on_2 + rem(n, 2)
    left_idx = Nx.iota({left_size}, names: [name], type: type)
    right_idx = {n_on_2} |> Nx.iota(names: [name], type: type) |> Nx.add(left_size)

    Nx.concatenate([
      Nx.multiply(left_idx, 2 / n),
      Nx.subtract(2, Nx.multiply(right_idx, 2 / n))
    ])
  end

  @doc """
  Triangular window

  See also: `bartlett/1`

  ## Options

    * `:n` - The window length. Mandatory option.
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.triangular(n: 3)
      #Nx.Tensor<
        f64[3]
        [0.5, 1.0, 0.5]
      >
  """
  defn triangular(opts \\ []) do
    transform(opts, fn opts ->
      opts = Keyword.validate!(opts, [:n, :name, type: {:f, 64}])

      n = opts[:n]
      name = opts[:name]
      type = opts[:type]

      unless n do
        raise "missing :n option"
      end

      triangular(n, name, type)
    end)
  end

  defp triangular(n, name, type) when rem(n, 2) == 1 do
    # odd case
    n_on_2 = div(n + 1, 2)

    idx = Nx.iota({n_on_2}, names: [name], type: type) |> Nx.add(1)

    left = Nx.multiply(idx, 2 / (n + 1))
    Nx.concatenate([left, left |> Nx.reverse() |> Nx.slice([1], [Nx.size(left) - 1])])
  end

  defp triangular(n, name, type) do
    # even case
    n_on_2 = div(n + 1, 2)

    idx = Nx.iota({n_on_2}, names: [name], type: type) |> Nx.add(1)

    left = Nx.multiply(idx, 2) |> Nx.subtract(1) |> Nx.divide(n)
    Nx.concatenate([left, Nx.reverse(left)])
  end

  @doc """
  Blackman window

  ## Options

    * `:n` - The window length. Mandatory option.
    * `:is_periodic` - If `true`, produces a periodic window,
       otherwise produces a symmetric window. Defaults to `true`
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.blackman(n: 5, is_periodic: false)
      #Nx.Tensor<
        f64[5]
        [-1.4901161193847656e-8, 0.34000001053081275, 0.9999999850988357, 0.34000001053081275, -1.4901161193847656e-8]
      >

      iex> NxSignal.Windows.blackman(n: 5, is_periodic: true)
      #Nx.Tensor<
        f64[5]
        [-1.4901161193847656e-8, 0.20077014493625245, 0.8492298742686417, 0.8492298742686417, 0.20077014493625245]
      >

      iex> NxSignal.Windows.blackman(n: 6, is_periodic: true)
      #Nx.Tensor<
        f64[6]
        [-1.4901161193847656e-8, 0.12999999636155424, 0.6300000210936009, 0.9999999850988357, 0.6300000210936009, 0.12999999636155424]
      >
  """
  defn blackman(opts \\ []) do
    transform(opts, fn opts ->
      opts = Keyword.validate!(opts, [:n, :name, is_periodic: true, type: {:f, 64}])

      n = opts[:n]
      name = opts[:name]
      type = opts[:type]
      is_periodic = opts[:is_periodic]

      unless n do
        raise "missing :n option"
      end

      blackman(n, name, type, is_periodic)
    end)
  end

  defp blackman(l, name, type, is_periodic) do
    import Nx.Defn.Kernel, only: [/: 2, *: 2, +: 2, -: 2]
    import Kernel, except: [/: 2, *: 2, +: 2, -: 2]

    l =
      if is_periodic do
        l + 1
      else
        l
      end

    m = ceil(l / 2)

    n = Nx.iota({m}, names: [name], type: type)

    left =
      0.42 - 0.5 * Nx.cos(2 * :math.pi() * n / (l - 1)) +
        0.08 * Nx.cos(4 * :math.pi() * n / (l - 1))

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
  Hamming window

  ## Options

    * `:n` - The window length. Mandatory option.
    * `:is_periodic` - If `true`, produces a periodic window,
       otherwise produces a symmetric window. Defaults to `true`
    * `:type` - the output type for the window. Defaults to `{:f, 32}`
    * `:name` - the axis name. Defaults to `nil`

  ## Examples

      iex> NxSignal.Windows.hamming(n: 5, is_periodic: true)
      #Nx.Tensor<
        f64[5]
        [0.08000001311302185, 0.3978522167650548, 0.9121478645310932, 0.9121478172561361, 0.39785214027257043]
      >
      iex> NxSignal.Windows.hamming(n: 5, is_periodic: false)
      #Nx.Tensor<
        f64[5]
        [0.08000001311302185, 0.5400000415649119, 1.0000000298023206, 0.5399999611359528, 0.0800000131130289]
      >
  """
  defn hamming(opts \\ []) do
    transform(opts, fn opts ->
      opts = Keyword.validate!(opts, [:n, :name, is_periodic: true, type: {:f, 64}])

      n = opts[:n]
      name = opts[:name]
      type = opts[:type]
      is_periodic = opts[:is_periodic]

      unless n do
        raise "missing :n option"
      end

      hamming(n, name, type, is_periodic)
    end)
  end

  defp hamming(l, name, type, is_periodic) do
    import Nx.Defn.Kernel, only: [/: 2, *: 2, +: 2, -: 2]
    import Kernel, except: [/: 2, *: 2, +: 2, -: 2]

    l =
      if is_periodic do
        l + 1
      else
        l
      end

    n = Nx.iota({l}, names: [name], type: type)

    window = 0.54 - 0.46 * Nx.cos(2 * :math.pi() * n / (l - 1))

    if is_periodic do
      Nx.slice(window, [0], [l - 1])
    else
      window
    end
  end
end
