.PHONY: install install-core test run benchmark predict api
install:        ; pip install -r requirements.txt
install-core:   ; pip install -r requirements-core.txt
test:           ; python -m unittest discover -s tests -v
run:            ; python main.py
benchmark:      ; python main.py --benchmark_balancing
predict:        ; python predict.py --input $(INPUT) --history $(HISTORY) --output pontuadas.csv
api:            ; uvicorn api:app --port 8000
