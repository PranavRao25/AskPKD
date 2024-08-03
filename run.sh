#!/bin/zsh

trap 'on_exit' SIGINT

on_exit() {
  rm -rf lancedb
  exit 0
}

streamlit run askpkd.py
wait $!
