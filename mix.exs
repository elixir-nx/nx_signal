defmodule NxSignal.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx_signal"
  @version "0.2.0"

  def project do
    [
      app: :nx_signal,
      version: @version,
      elixir: "~> 1.14",
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
        "guides/filtering.livemd",
        "guides/spectrogram.livemd"
      ],
      groups_for_extras: [
        Guides: Path.wildcard("guides/*.livemd")
      ],
      groups_for_functions: [
        "Functions: Time-Frequency": &(&1[:type] == :time_frequency),
        "Functions: Windowing": &(&1[:type] == :windowing),
        "Functions: Filters": &(&1[:type] == :filters),
        "Functions: Waveforms": &(&1[:type] == :waveforms)
      ]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.6"},
      {:ex_doc, "~> 0.29", only: :docs}
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
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.css" integrity="sha384-t5CR+zwDAROtph0PXGte6ia8heboACF9R5l/DiY+WZ3P2lxNgvJkQk5n7GPvLMYw" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.js" integrity="sha384-FaFLTlohFghEIZkw6VGwmf9ISTubWAVYW8tG8+w2LAIftJEULZABrF9PPFv+tVkH" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/contrib/auto-render.min.js" integrity="sha384-bHBqxz8fokvgoJ/sc17HODNxa42TlaEhB+w8ZJXTc2nZf1VgEaFZeZvT4Mznfz0v" crossorigin="anonymous"></script>
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
          ]
        });
      });
    </script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
