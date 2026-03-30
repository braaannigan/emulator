from __future__ import annotations

import streamlit as st

from src.viewer.training_metrics_viewer import render_training_metrics_page


def main(st_module=st) -> None:
    render_training_metrics_page(st_module)


if __name__ == "__main__":
    main()
