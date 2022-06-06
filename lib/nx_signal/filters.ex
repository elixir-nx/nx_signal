defmodule NxSignal.Filters do
  import Nx.Defn

  @pi :math.pi()

  defn sinc(t) do
    t = t * @pi
    zero_idx = Nx.equal(t, 0)

    # Define sinc(0) = 1
    Nx.select(zero_idx, 1, Nx.sin(t) / t)
  end
end
