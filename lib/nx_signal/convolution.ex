defmodule NxSignal.Convolution do
  @moduledoc """
  Convolution functions
  """

  import Nx.Defn

  deftransform convolve(in1, in2, opts \\ []) do
    {kernel, wrapped} = wrap_rank_zero(in2)
    {volume, ^wrapped} = wrap_rank_zero(in1)

    mode = Keyword.get(opts, :mode, "full")
    method = Keyword.get(opts, :method, "auto")

    axes =
      Nx.axes(kernel)

    kernel =
      Nx.reverse(kernel, axes: axes)

    kernel_shape =
      Nx.shape(kernel)
      |> Tuple.insert_at(0, 1)
      |> Tuple.insert_at(0, 1)

    kernel = Nx.reshape(kernel, kernel_shape)

    volume_shape =
      Nx.shape(volume)
      |> Tuple.insert_at(0, 1)
      |> Tuple.insert_at(0, 1)

    volume = Nx.reshape(volume, volume_shape)

    opts =
      case mode do
        "same" ->
          [padding: :same]

        "full" ->
          padding =
            Nx.shape(volume)
            |> Tuple.to_list()
            |> Enum.slice(2..-1)
            |> Enum.map(&(&1 - 1))
            |> Enum.map(&{&1, &1})

          [padding: padding]
      end

    Nx.conv(kernel, volume, opts)
    |> Nx.reverse()
    |> shape_output(mode, Nx.shape(volume))
    |> then(fn x -> Nx.reshape(x, drop_first_two(Nx.shape(x))) end)
    |> unwrap_rank_zero(wrapped)
  end

  def choose_conv_method(volume, kernel, opts \\ []) do
    v_shape = Nx.type(volume)
    k_shape = Nx.type(kernel)

    mode = Keyword.get(opts, :mode, "full")

    continue =
      if any_ints(v_shape) || any_ints(k_shape) do
        max_value = trunc(Nx.abs(Nx.reduce_max(volume))) * trunc(Nx.abs(Nx.reduce_max(kernel)))
        max_value = max_value * trunc(min(Nx.flat_size(volume), Nx.flat_size(kernel)))
        # Hard code mantissa bits
        if max_value > 2 ** 52 - 1 do
          "direct"
        end
      end

    case continue do
      nil ->
        fftconv_faster?(volume, kernel, mode)

      el ->
        el
    end
  end

  def fftconv_faster?(in1, in2, mode) do
  end

  def any_ints({inttype, _}) when inttype in [:u, :s], do: true
  def any_ints(_), do: false

  defp wrap_rank_zero(i) do
    case Nx.shape(i) do
      {} -> {Nx.reshape(i, {1}), true}
      _ -> {i, false}
    end
  end

  defp unwrap_rank_zero(tensor, true) do
    tensor[0]
  end

  defp unwrap_rank_zero(tensor, _) do
    tensor
  end

  defp drop_first_two(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.slice(2..-1)
    |> List.to_tuple()
  end

  defp shape_output(out, "full", _shape) do
    out
  end

  defp shape_output(out, "same", shape) do
    ac =
      shape
      |> Tuple.to_list()
      |> Enum.map(&(0..(&1 - 1)))

    out[ac]
  end
end
