defmodule NxSignal.Convolution do
  @moduledoc """
  Convolution functions
  """

  import Nx.Defn

  deftransform convolve(in1, in2, opts \\ []) do
    mode = Keyword.get(opts, :mode, "full")
    method = Keyword.get(opts, :method, "direct")

    input_rank =
      case {Nx.rank(in1), Nx.rank(in2)} do
        {0, r} ->
          r

        {r, 0} ->
          r

        {r, r} ->
          r

        {r1, r2} ->
          raise ArgumentError,
                "NxSignal.convolve/3 requires both inputs to have the same rank or one of them to be a scalar, got #{r1} and #{r2}"
      end

    kernel = Nx.reverse(in2)

    kernel_shape =
      case Nx.shape(kernel) do
        {} -> {1, 1, 1, 1}
        {n} -> {1, 1, 1, n}
        shape -> List.to_tuple([1, 1 | Tuple.to_list(shape)])
      end

    kernel = Nx.reshape(kernel, kernel_shape)

    volume_shape =
      case Nx.shape(in1) do
        {} -> {1, 1, 1, 1}
        {n} -> {1, 1, 1, n}
        shape -> List.to_tuple([1, 1 | Tuple.to_list(shape)])
      end

    volume = Nx.reshape(in1, volume_shape)

    opts =
      case mode do
        "same" ->
          kernel_spatial_shape =
            Nx.shape(kernel)
            |> Tuple.to_list()
            |> Enum.drop(2)

          padding =
            Enum.map(kernel_spatial_shape, fn k ->
              pad_total = k - 1
              # integer division for right side
              pad_right = div(pad_total, 2)
              # put the extra padding on the left
              pad_left = pad_total - pad_right
              {pad_left, pad_right}
            end)

          [padding: padding]

        "full" ->
          kernel_spatial_shape =
            Nx.shape(kernel)
            |> Tuple.to_list()
            |> Enum.drop(2)

          padding =
            Enum.map(kernel_spatial_shape, fn k ->
              {k - 1, k - 1}
            end)

          [padding: padding]
      end

    out =
      case method do
        "direct" ->
          Nx.conv(volume, kernel, opts)

        "fft" ->
          fftconvolve(volume, kernel, opts)
      end

    squeeze_axes =
      case input_rank do
        0 ->
          [0, 1, 2, 3]

        1 ->
          [0, 1, 2]

        _ ->
          [0, 1]
      end

    Nx.squeeze(out, axes: squeeze_axes)
  end

  def fftconvolve(volume, kernel, opts \\ []) do
    case {Nx.rank(volume), Nx.rank(kernel)} do
      {1, 1} ->
        Nx.product(volume, kernel)

      {a, b} when a == b ->
        s1 = Nx.shape(volume) |> Tuple.to_list()
        s2 = Nx.shape(kernel) |> Tuple.to_list()

        # Axes initialization
        axes = 0..(Nx.rank(volume) - 1)

        axes =
          for a <- axes, Enum.at(s1, a) != 1 || Enum.at(s2, a) == 1 do
            a
          end

        # Shape setup
        shape =
          for i <- 0..(length(s1) - 1) do
            if i not in axes do
              max(Enum.at(s1, i), Enum.at(s2, i))
            else
              Enum.at(s1, i) + Enum.at(s2, i) - 1
            end
          end

      # Frequency domain conversion

      _ ->
        raise ArgumentError, message: "Rank of volume and kernel must be equial."
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
end
