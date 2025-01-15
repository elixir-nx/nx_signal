defmodule NxSignal.Internal do
  @moduledoc false
  import Nx.Defn

  @omega 0.56714329040978387299997
  @expn1 0.36787944117144232159553

  defn lambert_w(z, k, tol \\ 1.0e-8) do
    rz = Nx.real(z)

    cond do
      Nx.is_infinity(rz) and rz > 0 ->
        z + 2.0 * Nx.Constants.pi() * k * Nx.Constants.i()

      Nx.is_infinity(rz) and rz < 0 ->
        -z + 2.0 * Nx.Constants.pi() * k * Nx.Constants.i()

      Nx.real(z) == 0 and Nx.imag(z) == 0 ->
        if k == 0 do
          z
        else
          z
          |> Nx.type()
          |> Nx.Type.to_real()
          |> Nx.Constants.neg_infinity()
        end

      Nx.equal(z, 1) and k == 0 ->
        @omega

      true ->
        halleys_method(z, k, tol)
    end
  end

  defnp halleys_method(z, k, tol) do
    absz = Nx.abs(z)

    w =
      cond do
        k == 0 ->
          cond do
            Nx.abs(z + @expn1) < 0.3 ->
              lambertw_branchpt(z)

            -1.0 < Nx.real(z) and Nx.real(z) < 1.5 and Nx.abs(Nx.imag(z)) < 1.0 and
                -2.5 * Nx.abs(Nx.imag(z)) - 0.2 < Nx.real(z) ->
              lambertw_pade0(z)

            true ->
              lambertw_asy(z, k)
          end

        k == -1 ->
          cond do
            absz <= @expn1 and Nx.imag(z) == 0.0 and Nx.real(z) < 0.0 ->
              Nx.log(-Nx.real(z))

            true ->
              lambertw_asy(z, k)
          end

        true ->
          lambertw_asy(z, k)
      end

    # Halley's Method
  end

  defnp lambertw_branchpt(z) do
    # TODO
  end

  defnp lambertw_pade0(z) do
    # TODO
  end

  defnp lambertw_asy(z, k) do
    # TODO
  end
end
