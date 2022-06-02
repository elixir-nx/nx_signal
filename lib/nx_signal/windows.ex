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
        f64[4]
        [0.25, 0.75, 0.75, 0.25]
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
end
