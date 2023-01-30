defmodule NxSignal.Filters do
  @moduledoc """
  Common filter functions.
  """
  import Nx.Defn

  @pi :math.pi()

  @doc ~S"""
  Calculates the normalized sinc function $sinc(t) = \frac{sin(\pi t)}{\pi t}$

  ## Examples

      iex> NxSignal.Filters.sinc(Nx.tensor([0, 0.25, 1]))
      #Nx.Tensor<
        f32[3]
        [1.0, 0.9003162980079651, -2.7827534054836178e-8]
      >
  """
  @doc type: :filters
  defn sinc(t) do
    t = t * @pi
    zero_idx = Nx.equal(t, 0)

    # Define sinc(0) = 1
    Nx.select(zero_idx, 1, Nx.sin(t) / t)
  end
end
