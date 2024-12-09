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

  def fftconv_faster?(x, y, mode) do
    {fft_opts, direct_opts} = _conv_opts(Nx.shape(x), Nx.shape(y), mode)

    offset =
      if Nx.shape(x) == 1 do
        -1.0e-3
      else
        -1.0e-4
      end

    constants =
      if Nx.rank(x) == 1 do
        %{
          "valid" => {1.89095737e-9, 2.1364985e-10, offset},
          "full" => {1.7649070e-9, 2.1414831e-10, offset},
          "same" =>
            if Nx.size(y) <= Nx.size(x) do
              {3.2646654e-9, 2.8478277e-10, offset}
            else
              {3.21635404e-9, 1.1773253e-8, -1.0e-5}
            end
        }
      else
        %{
          "valid" => {1.85927e-9, 2.11242e-8, offset},
          "full" => {1.99817e-9, 1.66174e-8, offset},
          "same" => {2.04735e-9, 1.55367e-8, offset}
        }
      end

    {fft0, direct0, offset0} = constants[mode]

    fft0 * fft_opts < direct0 * direct_opts + offset0
  end

  defp _conv_opts(x_shape, y_shape, mode) do
    out_shape =
      case mode do
        "full" ->
          for {n, k} <- Enum.zip(Tuple.to_list(x_shape), Tuple.to_list(y_shape)) do
            n + k - 1
          end

        "valid" ->
          for {n, k} <- Enum.zip(Tuple.to_list(x_shape), Tuple.to_list(y_shape)) do
            abs(n - k) + 1
          end

        "same" ->
          Tuple.to_list(x_shape)

        _ ->
          raise ArgumentError, message: "Unsupported mode: #{mode}"
      end

    s1 = x_shape |> Tuple.to_list()
    s2 = y_shape |> Tuple.to_list()

    direct_opts =
      if length(s1) == 1 do
        s1 = List.first(s1)
        s2 = List.first(s2)

        case mode do
          "full" ->
            s1 * s2

          "valid" ->
            if s2 >= s1 do
              (s2 - s1 + 1) * s1
            else
              (s1 - s2 + 1) * s2
            end

          "same" ->
            if s1 < s2 do
              s1 * s2
            else
              s1 * s2 - floor(s2 / 2) * floor((s2 + 1) / 2)
            end
        end
      else
        case mode do
          "full" ->
            min(_prod(s1), _prod(s2)) * _prod(out_shape)

          "valid" ->
            min(_prod(s1), _prod(s2)) * _prod(out_shape)

          "same" ->
            _prod(s1) * _prod(s2)
        end
      end

    full_out_shape =
      for {n, k} <- Enum.zip(Tuple.to_list(x_shape), Tuple.to_list(y_shape)) do
        n + k - 1
      end

    bigN = _prod(full_out_shape)
    fft_opts = 3 * bigN * :math.log(bigN)

    {fft_opts, direct_opts}
  end

  defp _prod(seq) when is_list(seq) do
    _prod(Nx.tensor(seq))
  end

  defp _prod(seq) do
    Nx.product(seq)
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
