defmodule NxSignal.Helpers do
  import ExUnit.Assertions

  @doc """
  Asserts `lhs` is close to `rhs`.
  """
  def assert_all_close(lhs, rhs, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    if Nx.all_close(lhs, rhs, atol: atol, rtol: rtol, equal_nan: opts[:equal_nan]) !=
         Nx.tensor(1, type: {:u, 8}) do
      flunk("""
      expected

      #{inspect(lhs)}

      to be within tolerance of

      #{inspect(rhs)}
      """)
    end
  end
end
