# NUSC-ISC-FootballAnalysis

- Todo list: [TODO.md](TODO.md)

- Reports:
  - 2026/04: Full report for ISC 2: [report-02.pdf](reports/2520/report-02.pdf)
  - 2026/03: Milestone 2 short write-up for ISC 2 (March 2026): [report-02b.pdf](reports/2520/report-02b.pdf)
  - 2026/02: Milestone 1 short write-up for ISC 2 (February 2026): [report-02a.pdf](reports/2520/report-02a.pdf)
  - 2025/12: Full report for ISC 1: [report-01.pdf](reports/2510/report-01.pdf)

- Export dependencies:
  ```bash
  conda env export --no-builds > environment.yml

  # If create
  conda env create -f environment.yml

  # If update
  conda env update -f envronment.yml --prune
  ```