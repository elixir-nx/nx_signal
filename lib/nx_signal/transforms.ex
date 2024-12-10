defmodule NxSignal.Transforms do
  import Nx.Defn

  deftransform fft_nd(tensor, opts \\ []) do
    axes = Keyword.get(opts, :axes, [-1])
    lengths = Keyword.get(opts, :lengths) || List.duplicate(nil, length(axes))

    Enum.zip_reduce(axes, lengths, tensor, fn axis, len, acc ->
      Nx.fft(acc, axis: axis, length: len)
    end)
  end

  deftransform ifft_nd(tensor, opts \\ []) do
    axes = Keyword.get(opts, :axes, [-1])
    lengths = Keyword.get(opts, :lengths) || List.duplicate(nil, length(axes))

    Enum.zip_reduce(axes, lengths, tensor, fn axis, len, acc ->
      Nx.ifft(acc, axis: axis, length: len)
    end)
  end
end
