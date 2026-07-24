defmodule NxSignal.Internal do
  @moduledoc false
  import Nx.Defn

  @omega 0.56714329040978387299997
  @expn1 0.36787944117144232159553

  deftransform lambert_w(z, k, opts \\ []) do
    opts = Keyword.validate!(opts, tol: 1.0e-8)

    z =
      if Nx.Type.complex?(Nx.type(z)) do
        Nx.as_type(z, :c128)
      else
        Nx.complex(Nx.as_type(z, :f64), 0)
      end

    lambert_w_n(z, k, tol: opts[:tol])
  end

  defnp lambert_w_n(z, k, opts) do
    tol = opts[:tol]
    rz = Nx.real(z)

    inf_pos = z + 2.0 * Nx.Constants.pi() * k * Nx.Constants.i()
    inf_neg = -z + 2.0 * Nx.Constants.pi() * k * Nx.Constants.i()
    main = halleys_method(z, k, tol)

    c_inf_pos = Nx.is_infinity(rz) and rz > 0
    c_inf_neg = Nx.is_infinity(rz) and rz < 0
    c_zero_k0 = z == 0 and k == 0
    c_zero = z == 0
    c_one_k0 = Nx.equal(z, 1) and k == 0

    Nx.select(
      c_inf_pos,
      inf_pos,
      Nx.select(
        c_inf_neg,
        inf_neg,
        Nx.select(
          c_zero_k0,
          z,
          Nx.select(c_zero, Nx.Constants.neg_infinity(:f64), Nx.select(c_one_k0, @omega, main))
        )
      )
    )
  end

  defnp halleys_method(z, k, tol) do
    rz = Nx.real(z)
    c_finite = not Nx.is_infinity(rz) and not Nx.is_nan(rz)

    safe_z = Nx.select(c_finite, z, 1.0)
    absz = Nx.abs(safe_z)

    branchpt_init = lambertw_branchpt(safe_z)
    pade0_init = lambertw_pade0(safe_z)
    asy_init = lambertw_asy(safe_z, k)
    log_init = Nx.log(-Nx.real(safe_z))

    c_bp = Nx.abs(safe_z + @expn1) < 0.3

    c_pade =
      -1.0 < Nx.real(safe_z) and Nx.real(safe_z) < 1.5 and
        Nx.abs(Nx.imag(safe_z)) < 1.0 and
        -2.5 * Nx.abs(Nx.imag(safe_z)) - 0.2 < Nx.real(safe_z)

    c_k_neg1_special =
      k == -1 and absz <= @expn1 and Nx.imag(safe_z) == 0.0 and Nx.real(safe_z) < 0.0

    k0 = Nx.select(c_bp, branchpt_init, Nx.select(c_pade, pade0_init, asy_init))
    w = Nx.select(k == 0, k0, Nx.select(c_k_neg1_special, log_init, asy_init))

    {w, _} =
      while {w, {safe_z, tol, i = 0}}, i < 100 do
        ew_neg = Nx.exp(-w)
        wewz_neg = w - safe_z * ew_neg
        wn_neg = w - wewz_neg / (w + 1.0 - (w + 2.0) * wewz_neg / (2.0 * w + 2.0))

        ew_pos = Nx.exp(w)
        wew = w * ew_pos
        wewz_pos = wew - safe_z
        wn_pos = w - wewz_pos / (wew + ew_pos - (w + 2.0) * wewz_pos / (2.0 * w + 2.0))

        wn = Nx.select(Nx.real(w) >= 0, wn_neg, wn_pos)
        {wn, {safe_z, tol, i + 1}}
      end

    w
  end

  defnp lambertw_branchpt(z) do
    m_e =
      Nx.Constants.e()

    p = Nx.sqrt(2.0 * (m_e * z + 1.0))

    cevalpoly_2(p, -1.0 / 3.0, 1.0, -1.0)
  end

  defnp lambertw_pade0(z) do
    z * cevalpoly_2(z, 12.85106382978723404255, 12.34042553191489361902, 1.0) /
      cevalpoly_2(z, 32.53191489361702127660, 14.34042553191489361702, 1.0)
  end

  defnp lambertw_asy(z, k) do
    w =
      Nx.log(z) + 2.0 * Nx.Constants.pi() * k * Nx.Constants.i()

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
