defmodule NxSignal.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx-signal"
  @version "0.5.0"

  def project do
    [
      app: :nx_signal,
      version: @version,
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      docs: docs(),
      name: "NxSignal",
      description: "Digital Signal Processing extension for Nx",
      package: package(),
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ]
    ]
  end

  defp elixirc_paths(:test), do: ["test/support", "lib"]
  defp elixirc_paths(_), do: ["lib"]

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp docs do
    [
      main: "NxSignal",
      source_url_pattern: "#{@source_url}/blob/v#{@version}/nx-signal/%{path}#L%{line}",
      before_closing_body_tag: &before_closing_body_tag/1,
      extras: [
        # "guides/intro-to-nx-signal.livemd"
      ],
      groups_for_functions: [
        "Functions: Time-Frequency": &(&1[:type] == :time_frequency),
        "Functions: Windowing": &(&1[:type] == :windowing),
        "Functions: Filters": &(&1[:type] == :filters)
      ]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.4", github: "elixir-nx/nx", sparse: "nx"},
      {:ex_doc, "~> 0.29.0", only: :docs}
    ]
  end

  defp package do
    [
      maintainers: ["Paulo Valente"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp before_closing_body_tag(:html) do
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.19/dist/katex.min.css" integrity="sha384-beuqjL2bw+6DBM2eOpr5+Xlw+jiH44vMdVQwKxV28xxpoInPHTVmSvvvoPq9RdSh" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.19/dist/katex.min.js" integrity="sha384-aaNb715UK1HuP4rjZxyzph+dVss/5Nx3mLImBe9b0EW4vMUkc1Guw4VRyQKBC0eG" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.19/dist/contrib/auto-render.min.js" integrity="sha384-+XBljXPPiv+OzfbB3cVmLHf4hdUFHlWNZN5spNQ7rmHTXpd7WvJum6fIACpNNfIR" crossorigin="anonymous"
            onload="renderMathInElement(document.body);"></script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
