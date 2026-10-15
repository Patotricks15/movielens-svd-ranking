.PHONY: all clean docker tables

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

tables:
	cd .. && python scripts/export_results_table.py && python scripts/export_feature_tables.py

docker:
	docker run --rm -v "$(CURDIR)":/work -w /work texlive/texlive:latest \
		latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C
