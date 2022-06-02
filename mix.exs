defmodule NxSignal.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_signal,
      version: "0.1.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.2", github: "elixir-nx/nx", branch: "main", sparse: "nx", override: true},
      {:torchx, "~> 0.1", github: "elixir-nx/nx", branch: "main", sparse: "torchx"}
    ]
  end
end
