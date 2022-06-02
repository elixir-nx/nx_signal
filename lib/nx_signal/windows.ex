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
end
