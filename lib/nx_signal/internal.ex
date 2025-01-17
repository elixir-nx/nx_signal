defmodule NxSignal.Internal do
  @moduledoc false
  import Nx.Defn

  @omega 0.56714329040978387299997
  @expn1 0.36787944117144232159553

  defn lambert_w(z, k, opts \\ []) do
    opts = keyword!(opts, tol: 1.0e-8)
    tol = Nx.f64(opts[:tol])
    # z = Nx.as_type(z, z |> Nx.type() |> Nx.Type.to_complex())
    z = Nx.as_type(z, {:c, 128})
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
        Nx.f64(@omega)

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
            Nx.abs(z + Nx.f64(@expn1)) < 0.3 ->
              lambertw_branchpt(z)

            -1.0 < Nx.real(z) and Nx.real(z) < 1.5 and Nx.abs(Nx.imag(z)) < 1.0 and
                -2.5 * Nx.abs(Nx.imag(z)) - 0.2 < Nx.real(z) ->
              lambertw_pade0(z)

            true ->
              lambertw_asy(z, k)
          end

        k == -1 ->
          cond do
            absz <= Nx.f64(@expn1) and Nx.imag(z) == 0.0 and Nx.real(z) < 0.0 ->
              Nx.log(-Nx.real(z))

            true ->
              lambertw_asy(z, k)
          end

        true ->
          lambertw_asy(z, k)
      end

    # Halley's Method
    cond do
      Nx.real(w) >= 0 ->
        {w, _} =
          while {w, {z, tol, i = 0}}, i < 100 do
            ew = Nx.exp(-w)
            wewz = w - z * ew
            wn = w - wewz / (w + 1.0 - (w + 2.0) * wewz / (2.0 * w + 2.0))

            if Nx.abs(wn - w) <= tol * Nx.abs(wn) do
              {wn, {z, tol, 100}}
            else
              {wn, {z, tol, i + 1}}
            end
          end

        w

      true ->
        {w, _} =
          while {w, {z, tol, i = 0}}, i < 100 do
            ew = Nx.exp(w)
            wew = w * ew
            wewz = wew - z
            wn = w - wewz / (wew + ew - (w + 2.0) * wewz / (2.0 * w + 2.0))

            if Nx.abs(wn - w) <= tol * Nx.abs(wn) do
              {wn, {z, tol, 100}}
            else
              {wn, {z, tol, i + 1}}
            end
          end

        w
    end
  end

  defnp lambertw_branchpt(z) do
    m_e =
      Nx.Constants.e({:f, 64})

    p = Nx.sqrt(2.0 * (m_e * z + 1.0))

    cevalpoly_2(p, Nx.f64(-1.0 / 3.0), Nx.f64(1.0), Nx.f64(-1.0))
  end

  defnp lambertw_pade0(z) do
    z * cevalpoly_2(z, Nx.f64(12.85106382978723404255), Nx.f64(12.34042553191489361902), 1.0) /
      cevalpoly_2(z, Nx.f64(32.53191489361702127660), Nx.f64(14.34042553191489361702), 1.0)
  end

  defnp lambertw_asy(z, k) do
    w = Nx.log(z) + 2.0 * Nx.Constants.pi(:f64) * k * Nx.Constants.i()
    w - Nx.log(w)
  end

  defnp cevalpoly_2(z, c0, c1, c2) do
    s = Nx.abs(z) ** 2
    r = 2 * Nx.real(z)
    b = -s * c0 + c2
    a = r * c0 + c1
    z * a + b
  end
end
