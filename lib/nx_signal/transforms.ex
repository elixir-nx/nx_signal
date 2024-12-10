defmodule NxSignal.Transforms do
  import Nx.Defn

  deftransform fft_nd(tensor, opts \\ []) do
    Enum.reduce(Keyword.get(opts, :axes, [-1]), tensor, fn el, acc ->
      Nx.fft(acc, axis: el)
    end)
  end

  deftransform ifft_nd(tensor, opts \\ []) do
    Enum.reduce(Keyword.get(opts, :axes, [-1]), tensor, fn el, acc ->
      Nx.ifft(acc, axis: el)
    end)
  end
end
