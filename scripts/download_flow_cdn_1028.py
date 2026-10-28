#!/usr/bin/env python3
"""One-time Flow Planner download recovery, issued 2026-10-28.

The embedded public CDN URL expires at EXPIRES_UTC. It bypasses the initial
Hugging Face resolver only; it is not a permanent mirror and cannot guarantee
that the evaluation node can reach the CDN. Already verified local assets work
without the URL, including after expiry. The embedded configuration is byte-for-
byte the original official model_config.yaml, not a regenerated configuration.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.parse


SOURCE_REPOSITORY = "https://github.com/DiffusionAD/Flow-Planner"
MODEL_REPOSITORY = "https://huggingface.co/ttwhy/flow-planner"
REVISION = "b627934a8db3021892afd13f5278ba0149532efc"
EXPIRES_EPOCH = 1793178095
EXPIRES_UTC = "2026-10-28T09:01:35Z"
DIRECT_MODEL_URL = 'https://us.aws.cdn.hf.co/xet-bridge-us/69b43718fe541d1316e9dd27/8ace7cdf92a4751f84f40ff91d564cb22ed5e0b4fced86ef4d009eea18e81822?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27model.pth%3B+filename%3D%22model.pth%22%3B&xip=y5XLYu9WKow&X-Xet-Cas-Uid=public&user_id=public&Expires=1793178095&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly91cy5hd3MuY2RuLmhmLmNvL3hldC1icmlkZ2UtdXMvNjliNDM3MThmZTU0MWQxMzE2ZTlkZDI3LzhhY2U3Y2RmOTJhNDc1MWY4NGY0MGZmOTFkNTY0Y2IyMmVkNWUwYjRmY2VkODZlZjRkMDA5ZWVhMThlODE4MjJcXD9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPWF0dGFjaG1lbnQlM0IrZmlsZW5hbWUlMkElM0RVVEYtOCUyNyUyN21vZGVsLnB0aCUzQitmaWxlbmFtZSUzRCUyMm1vZGVsLnB0aCUyMiUzQiZ4aXA9eTVYTFl1OVdLb3cmWC1YZXQtQ2FzLVVpZD1wdWJsaWMmdXNlcl9pZD1wdWJsaWMiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkVwb2NoVGltZSI6MTc5MzE3ODA5NX19fV19&Signature=MEYCIQCneuSIuKLSIzAKfPqojs6IjTRQ%7ESAajVdBhemWdWXX6gIhAKIWa4x54zUtmTQaERyI8qu%7Eq8Wr6Dxd91Hmc1kUjYOp&Key-Pair-Id=01KXEF4KZ1B6FV465MAWR4M21F&Hash-Algorithm=SHA256'
ASSETS = {
    "model.pth": {
        "size_bytes": 57244066,
        "sha256": "9231b4c02968c9e42dd2881ad3e7dd4bd362796f7b2c0158e0458c4ab3bde88e",
    },
    "model_config.yaml": {
        "size_bytes": 7036,
        "sha256": "2914c4f174e7fba8baf9ee7d42300c640a285f9d6fa13d36b78f0a3cbb9a03fe",
    },
}
# Original published YAML bytes; retaining them avoids another HF request.
CONFIG_B64 = (
    'bW9kZWw6CiAgX3RhcmdldF86IGZsb3dfcGxhbm5lci5tb2RlbC5mbG93X3BsYW5uZXJfbW9kZWwuZmxvd19wbGFubmVyLkZsb3dQ'
    'bGFubmVyCiAgbmVpZ2hib3JfbnVtOiAzMgogIG5laWdoYm9yX3ByZWRfbnVtOiAwCiAgc3RhdGljX251bTogNQogIGxhbmVfbnVt'
    'OiA3MAogIGNmZ19uZWlnaGJvcl9udW06IDEwCiAgY2ZnX3Byb2I6IDAuMwogIGNmZ193ZWlnaHQ6IDEuMCAjIGRpc2FibGVkIGJ5'
    'IGRlZmF1bHQKICBjZmdfdHlwZTogbmVpZ2hib3JzCiAgZnV0dXJlX2xlbjogODAKICBhY3Rpb25fbGVuOiAyMAogIGFjdGlvbl9v'
    'dmVybGFwOiAxMAogIHN0YXRlX2RpbTogNAogIG1vZGVsX3R5cGU6IHhfc3RhcnQKICBmbG93X29kZToKICAgIF90YXJnZXRfOiBm'
    'bG93X3BsYW5uZXIubW9kZWwuZmxvd19wbGFubmVyX21vZGVsLmZsb3dfdXRpbHMuZmxvd19vZGUuRmxvd09ERQogICAgcGF0aDoK'
    'ICAgICAgX3RhcmdldF86IGZsb3dfbWF0Y2hpbmcucGF0aC5hZmZpbmUuQWZmaW5lUHJvYlBhdGgKICAgICAgc2NoZWR1bGVyOgog'
    'ICAgICAgIF90YXJnZXRfOiBmbG93X21hdGNoaW5nLnBhdGguc2NoZWR1bGVyLkNvbmRPVFNjaGVkdWxlcgogICAgdGltZV9zYW1w'
    'bGVyOgogICAgICBfdGFyZ2V0XzogZmxvd19wbGFubmVyLm1vZGVsLm1vZGVsX3V0aWxzLnRpbWVfc2FtcGxlci5UaW1lU2FtcGxl'
    'cgogICAgICBzYW1wbGVfbWV0aG9kOiB1bmlmb3JtCiAgICAgIGRldmljZTogJHtkZXZpY2V9CiAgICAgIGVwczogMC4wMDEKICAg'
    'ICAgYWxwaGE6IDEuMAogICAgICBiZXRhOiAxLjUKICAgIGNmZ193ZWlnaHQ6ICR7Li5jZmdfd2VpZ2h0fQogICAgc2FtcGxlX3N0'
    'ZXBzOiA0CiAgICBzYW1wbGVfbWV0aG9kOiBtaWRwb2ludAogICAgc2FtcGxlX3RlbXBlcmF0dXJlOiAxLjAKICBhc3NlbWJsZV9t'
    'ZXRob2Q6IGF2ZXJhZ2UKICBvYnNfbm9ybWFsaXplcjoKICAgIF90YXJnZXRfOiBmbG93X3BsYW5uZXIuZGF0YS5ub3JtYWxpemF0'
    'aW9uLm9ic19ub3JtYWxpemUuT2JzZXJ2YXRpb25Ob3JtYWxpemVyCiAgICBjb25maWc6ICR7bm9ybWFsaXphdGlvbl9zdGF0c30K'
    'ICBzdGF0ZV9ub3JtYWxpemVyOgogICAgX3RhcmdldF86IGZsb3dfcGxhbm5lci5kYXRhLm5vcm1hbGl6YXRpb24uc3RhdGVfbm9y'
    'bWFsaXplLlN0YXRlTm9ybWFsaXplcgogICAgY29uZmlnOiAke25vcm1hbGl6YXRpb25fc3RhdHN9CiAgICBmdXR1cmVfZG93bnNh'
    'bXBsaW5nX21ldGhvZDogJHtkYXRhLmRhdGFzZXQudHJhaW4uZnV0dXJlX2Rvd25zYW1wbGluZ19tZXRob2R9CiAgICBwcmVkaWN0'
    'ZWRfbmVpZ2hib3JfbnVtOiAke2RhdGEuZGF0YXNldC50cmFpbi5wcmVkaWN0ZWRfbmVpZ2hib3JfbnVtfQogIGRhdGFfcHJvY2Vz'
    'c29yOgogICAgX3RhcmdldF86IGZsb3dfcGxhbm5lci5tb2RlbC5tb2RlbF91dGlscy5pbnB1dF9wcmVwcm9jZXNzLk1vZGVsSW5w'
    'dXRQcm9jZXNzb3IKICAgIGZ1dHVyZV9sZW46ICR7Li5mdXR1cmVfbGVufQogICAgb2JzX25vcm1hbGl6ZXI6ICR7Li5vYnNfbm9y'
    'bWFsaXplcn0KICAgIHN0YXRlX25vcm1hbGl6ZXI6ICR7Li5zdGF0ZV9ub3JtYWxpemVyfQogICAgbmVpZ2hib3JfcHJlZF9udW06'
    'ICR7Li5uZWlnaGJvcl9wcmVkX251bX0KICBtb2RlbF9lbmNvZGVyOgogICAgX3RhcmdldF86IGZsb3dfcGxhbm5lci5tb2RlbC5m'
    'bG93X3BsYW5uZXJfbW9kZWwuZW5jb2Rlci5GbG93UGxhbm5lckVuY29kZXIKICAgIGVuY29kZXJfaGlkZGVuX2RpbTogMTkyCiAg'
    'ICB3aXRoX2Vnb19oaXN0b3J5OiBmYWxzZQogICAgbmVpZ2hib3JfZW5jb2RlcjoKICAgICAgX3RhcmdldF86IGZsb3dfcGxhbm5l'
    'ci5tb2RlbC5tb2R1bGVzLmVuY29kZXJfbW9kdWxlcy5BZ2VudEZ1c2lvbkVuY29kZXIKICAgICAgcGFzdF90aW1lX2xlbjogMjEK'
    'ICAgICAgZHJvcF9wYXRoX3JhdGU6IDAuMwogICAgICBoaWRkZW5fZGltOiAkey4uZW5jb2Rlcl9oaWRkZW5fZGltfQogICAgICBs'
    'YXllcl9udW06IDMKICAgICAgdG9rZW5zX21scF9kaW06IDY0CiAgICAgIGNoYW5uZWxzX21scF9kaW06IDEyOAogICAgc3RhdGlj'
    'X2VuY29kZXI6CiAgICAgIF90YXJnZXRfOiBmbG93X3BsYW5uZXIubW9kZWwubW9kdWxlcy5lbmNvZGVyX21vZHVsZXMuU3RhdGlj'
    'RnVzaW9uRW5jb2RlcgogICAgICBzdGF0aWNfb2JqZWN0c19zdGF0ZV9kaW06IDEwCiAgICAgIGRyb3BfcGF0aF9yYXRlOiAwLjMK'
    'ICAgICAgaGlkZGVuX2RpbTogJHsuLmVuY29kZXJfaGlkZGVuX2RpbX0KICAgIGxhbmVfZW5jb2RlcjoKICAgICAgX3RhcmdldF86'
    'IGZsb3dfcGxhbm5lci5tb2RlbC5tb2R1bGVzLmVuY29kZXJfbW9kdWxlcy5MYW5lRnVzaW9uRW5jb2RlcgogICAgICBsYW5lX3Bv'
    'aW50c19udW06IDIwCiAgICAgIGRyb3BfcGF0aF9yYXRlOiAwLjMKICAgICAgaGlkZGVuX2RpbTogJHsuLmVuY29kZXJfaGlkZGVu'
    'X2RpbX0KICAgICAgbGF5ZXJfbnVtOiAzCiAgICAgIHRva2Vuc19tbHBfZGltOiA2NAogICAgICBjaGFubmVsc19tbHBfZGltOiAx'
    'MjgKICAgIHJvdXRlX2VuY29kZXI6CiAgICAgIF90YXJnZXRfOiBmbG93X3BsYW5uZXIubW9kZWwubW9kdWxlcy5lbmNvZGVyX21v'
    'ZHVsZXMuUm91dGVFbmNvZGVyCiAgICAgIHJvdXRlX251bTogMjUKICAgICAgcm91dGVfcG9pbnRzX251bTogMjAKICAgICAgZHJv'
    'cF9wYXRoX3JhdGU6IDAuMwogICAgICBoaWRkZW5fZGltOiAke21vZGVsLm1vZGVsX2RlY29kZXIuaGlkZGVuX2RpbX0KICAgICAg'
    'dG9rZW5zX21scF9kaW06IDMyCiAgICAgIGNoYW5uZWxzX21scF9kaW06IDY0CiAgICBhY3Rpb25fbGVuZ3RoOiAkey4uYWN0aW9u'
    'X2xlbn0KICAgIGFjdGlvbl9vdmVybGFwOiAkey4uYWN0aW9uX292ZXJsYXB9CiAgICBzdGF0aWNfb2JqZWN0c19udW06IDUKICAg'
    'IGxhbmVfbnVtOiA3MAogICAgbGFuZV9kaW06IDEyCiAgICBuZWlnaGJvcl9hZ2VudF9udW06IDMyCiAgICBuZWlnaGJvcl9wcmVk'
    'X251bTogJHsuLm5laWdoYm9yX3ByZWRfbnVtfQogIG1vZGVsX2RlY29kZXI6CiAgICBfdGFyZ2V0XzogZmxvd19wbGFubmVyLm1v'
    'ZGVsLmZsb3dfcGxhbm5lcl9tb2RlbC5kZWNvZGVyLkZsb3dQbGFubmVyRGVjb2RlcgogICAgaGlkZGVuX2RpbTogMjU2CiAgICB0'
    'X2VtYmVkZGVyOgogICAgICBfdGFyZ2V0XzogZmxvd19wbGFubmVyLm1vZGVsLm1vZHVsZXMuZGVjb2Rlcl9tb2R1bGVzLlRpbWVz'
    'dGVwRW1iZWRkZXIKICAgICAgaGlkZGVuX3NpemU6ICR7bW9kZWwubW9kZWxfZGVjb2Rlci5oaWRkZW5fZGltfQogICAgZGVwdGg6'
    'IDQKICAgIGFnZW50c19oaWRkZW5fZGltOiAkey4ubW9kZWxfZW5jb2Rlci5lbmNvZGVyX2hpZGRlbl9kaW19CiAgICBsYW5lX2hp'
    'ZGRlbl9kaW06ICR7Li5tb2RlbF9lbmNvZGVyLmVuY29kZXJfaGlkZGVuX2RpbX0KICAgIGhlYWRzOiA4CiAgICBlbmFibGVfYXR0'
    'bl9kaXN0OiB0cnVlCiAgICBhY3RfcGVfdHlwZTogZml4ZWRfc2luCiAgICBkZXZpY2U6ICR7ZGV2aWNlfQogICAgbmVpZ2hib3Jf'
    'bnVtOiAkey4ubmVpZ2hib3JfbnVtfQogICAgbmVpZ2hib3JfcHJlZF9udW06ICR7Li5uZWlnaGJvcl9wcmVkX251bX0KICAgIHN0'
    'YXRpY19udW06ICR7Li5zdGF0aWNfbnVtfQogICAgbGFuZV9udW06ICR7Li5sYW5lX251bX0KICAgIGNmZ19uZWlnaGJvcl9udW06'
    'ICR7Li5jZmdfbmVpZ2hib3JfbnVtfQogICAgZnV0dXJlX2xlbjogJHsuLmZ1dHVyZV9sZW59CiAgICBhY3Rpb25fbGVuOiAkey4u'
    'YWN0aW9uX2xlbn0KICAgIGFjdGlvbl9vdmVybGFwOiAkey4uYWN0aW9uX292ZXJsYXB9CiAgICBzdGF0ZV9kaW06ICR7Li5zdGF0'
    'ZV9kaW19CmNvcmU6CiAgX3RhcmdldF86IGZsb3dfcGxhbm5lci5jb3JlLmZsb3dfbWF0Y2hpbmdfY29yZS5GbG93TWF0Y2hpbmdD'
    'b3JlCiAgZGV2aWNlOiAke2RldmljZX0KICBpbnB1dF9hdWc6CiAgICBfdGFyZ2V0XzogZmxvd19wbGFubmVyLmRhdGEuYXVnbWVu'
    'dGF0aW9uLm51cGxhbl9hdWdtZW50YXRpb24uTnVQbGFuQXVnbWVudAogICAgc3RhdGVfYXVnOgogICAgICBfdGFyZ2V0XzogZmxv'
    'd19wbGFubmVyLmRhdGEuYXVnbWVudGF0aW9uLnN0YXRlX2F1Zy5TdGF0ZVBlcnR1cmJhdGlvbgogICAgICBhdWdtZW50X3Byb2I6'
    'IDAuNQogICAgbWFwX2F1ZzogbnVsbAogICAgZGV2aWNlOiAke2RldmljZX0KICBlZ29fcGxhbm5pbmdfbG9zczogMS4wCiAgY29u'
    'c2lzdGVuY3lfbG9zczogMC41Cm5vcm1hbGl6YXRpb25fc3RhdHM6CiAgZWdvOgogICAgbG9nOiB7fQogICAgdW5pZm9ybToKICAg'
    'ICAgbWVhbjoKICAgICAgLSAxMAogICAgICAtIDAKICAgICAgLSAwCiAgICAgIC0gMAogICAgICBzdGQ6CiAgICAgIC0gMjAKICAg'
    'ICAgLSAyMAogICAgICAtIDEKICAgICAgLSAxCiAgbmVpZ2hib3I6CiAgICBsb2c6IHt9CiAgICB1bmlmb3JtOgogICAgICBtZWFu'
    'OgogICAgICAtIDEwCiAgICAgIC0gMAogICAgICAtIDAKICAgICAgLSAwCiAgICAgIHN0ZDoKICAgICAgLSAyMAogICAgICAtIDIw'
    'CiAgICAgIC0gMQogICAgICAtIDEKICBlZ29fcGFzdDoKICAgIG1lYW46CiAgICAtIDEwCiAgICAtIDAKICAgIC0gMAogICAgLSAw'
    'CiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAg'
    'ICAtIDAKICAgIHN0ZDoKICAgIC0gMjAKICAgIC0gMjAKICAgIC0gMQogICAgLSAxCiAgICAtIDIwCiAgICAtIDIwCiAgICAtIDIw'
    'CiAgICAtIDIwCiAgICAtIDIwCiAgICAtIDIwCiAgICAtIDEKICAgIC0gMQogICAgLSAxCiAgICAtIDEKICBlZ29fY3VycmVudDoK'
    'ICAgIG1lYW46CiAgICAtIDEwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICAtIDUKICAgIC0gMAogICAgLSAwCiAgICAtIDAK'
    'ICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICBzdGQ6CiAg'
    'ICAtIDIwCiAgICAtIDIwCiAgICAtIDEKICAgIC0gMQogICAgLSA1CiAgICAtIDEKICAgIC0gMQogICAgLSAxCiAgICAtIDEKICAg'
    'IC0gMQogICAgLSAyMAogICAgLSAyMAogICAgLSAxCiAgICAtIDEKICAgIC0gMQogICAgLSAxCiAgbmVpZ2hib3JfcGFzdDoKICAg'
    'IG1lYW46CiAgICAtIDEwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAg'
    'IC0gMAogICAgLSAwCiAgICAtIDAKICAgIHN0ZDoKICAgIC0gMjAKICAgIC0gMjAKICAgIC0gMQogICAgLSAxCiAgICAtIDIwCiAg'
    'ICAtIDIwCiAgICAtIDIwCiAgICAtIDIwCiAgICAtIDEKICAgIC0gMQogICAgLSAxCiAgbWFwX29iamVjdHM6CiAgICBtZWFuOgog'
    'ICAgLSAxMAogICAgLSAwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAg'
    'IC0gMAogICAgc3RkOgogICAgLSAyMAogICAgLSAyMAogICAgLSAxCiAgICAtIDEKICAgIC0gMjAKICAgIC0gMjAKICAgIC0gMQog'
    'ICAgLSAxCiAgICAtIDEKICAgIC0gMQogIGxhbmVzOgogICAgbWVhbjoKICAgIC0gMTAKICAgIC0gMAogICAgLSAwCiAgICAtIDAK'
    'ICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICBzdGQ6CiAg'
    'ICAtIDIwCiAgICAtIDIwCiAgICAtIDIwCiAgICAtIDIwCiAgICAtIDIwCiAgICAtIDIwCiAgICAtIDIwCiAgICAtIDIwCiAgICAt'
    'IDEKICAgIC0gMQogICAgLSAxCiAgICAtIDEKICBsYW5lc19zcGVlZGxpbWl0OgogICAgbWVhbjoKICAgIC0gMAogICAgc3RkOgog'
    'ICAgLSAyMAogIHJvdXRlczoKICAgIG1lYW46CiAgICAtIDEwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAgIC0g'
    'MAogICAgLSAwCiAgICAtIDAKICAgIC0gMAogICAgLSAwCiAgICAtIDAKICAgIC0gMAogICAgc3RkOgogICAgLSAyMAogICAgLSAy'
    'MAogICAgLSAyMAogICAgLSAyMAogICAgLSAyMAogICAgLSAyMAogICAgLSAyMAogICAgLSAyMAogICAgLSAxCiAgICAtIDEKICAg'
    'IC0gMQogICAgLSAxCiAgcm91dGVzX3NwZWVkbGltaXQ6CiAgICBtZWFuOgogICAgLSAwCiAgICBzdGQ6CiAgICAtIDIwCnJlY29y'
    'ZGVyOgogIHRlbnNvcmJvYXJkOgogICAgX3RhcmdldF86IGZsb3dfcGxhbm5lci5yZWNvcmRlci50ZW5zb3Jib2FyZC5UZW5zb3Ji'
    'b2FyZFJlY29yZGVyCiAgICBzYXZlX3BhdGg6ICR7b2MuZW52OlRFTlNPUkJPQVJEX0xPR19QQVRIfQogICAgcmFuazogJHtvYy5l'
    'bnY6UkFOS30Kb3B0aW1pemVyOgogIF90YXJnZXRfOiB0b3JjaC5vcHRpbS5BZGFtVwogIF9jb252ZXJ0XzogYWxsCiAgbHI6IDAu'
    'MDAwMQpzY2hlZHVsZXI6CiAgX3RhcmdldF86IGZsb3dfcGxhbm5lci5jb3JlLmNvbW1vbi50cmFpbl9zY2hlZHVsZXIuQ29zaW5l'
    'QW5uZWFsaW5nV2FybVVwUmVzdGFydHMKICBlcG9jaDogJHt0cmFpbi5lcG9jaH0KICB3YXJtX3VwX2Vwb2NoOiA1CmVtYToKICBf'
    'dGFyZ2V0XzogdGltbS51dGlscy5Nb2RlbEVtYQogIGRlY2F5OiAwLjk5OQogIGRldmljZTogJHtkZXZpY2V9CnByb2plY3Rfcm9v'
    'dDogJHtvYy5lbnY6UFJPSkVDVF9ST09UfQpzYXZlX2RpcjogJHtvYy5lbnY6U0FWRV9ESVJ9L291dHB1dHMvJHtwcm9qZWN0fS8k'
    'e2pvYl9uYW1lfS8ke25vdzolWS0lbS0lZF8lSC0lTS0lU30Kc2F2ZV9ldmVyeV9zaW5jZTogMApyZXN1bWVfcGF0aDogbnVsbApw'
    'cmV0cmFpbmVkX2NoZWNrcG9pbnQ6IG51bGwKcHJvamVjdDogRmxvd1BsYW5uZXJUcmFpbmluZwpqb2JfbmFtZTogZmxvd19wbGFu'
    'bmVyCmxvZ2dlcl9sZXZlbDogaW5mbwpzZWVkOiAzNDAyCmRldmljZTogY3VkYQpudW1fd29ya2VyczogNAp1c2Vfd2FuZGI6IGZh'
    'bHNlCnBpbl9tZW06IHRydWUKdHJhaW46CiAgYmF0Y2hfc2l6ZTogMjA0OAogIGVwb2NoOiA0MDAKICBzYXZlX3V0ZDogMjAKZGRw'
    'OgogIGRpc3RyaWJ1dGVkOiB0cnVlCiAgd29ybGRfc2l6ZTogJHtvYy5lbnY6V09STERfU0laRX0KICByYW5rOiAke29jLmVudjpM'
    'T0NBTF9SQU5LfQogIGluaXRfcHJvY2Vzc19ncm91cDogbmNjbAogIGFkZHI6ICR7b2MuZW52Ok1BU1RFUl9BRERSLCBsb2NhbGhv'
    'c3R9CiAgcG9ydDogJHtvYy5lbnY6TUFTVEVSOlBPUlQsIDI5NTI5fQptb2RlbF90eXBlOiB4X3N0YXJ0Cg=='
)


def verified(path, asset):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != asset["size_bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == asset["sha256"]


def publish(source, destination, asset):
    """Never overwrite a destination that another process or user created."""
    if not verified(source, asset):
        raise ValueError(f"Size/SHA256 mismatch; refusing to publish {destination.name}")
    try:
        os.link(source, destination)
    except FileExistsError:
        if not verified(destination, asset):
            raise ValueError(f"Existing mismatched file preserved: {destination}")


def download_model(destination):
    parsed = urllib.parse.urlsplit(DIRECT_MODEL_URL)
    if parsed.scheme != "https" or parsed.hostname != "us.aws.cdn.hf.co":
        raise ValueError("Unexpected direct CDN endpoint")
    if int(urllib.parse.parse_qs(parsed.query)["Expires"][0]) != EXPIRES_EPOCH:
        raise ValueError("CDN URL expiration does not match the recorded timestamp")
    if time.time() >= EXPIRES_EPOCH:
        raise RuntimeError(
            f"This one-time CDN URL expired at {EXPIRES_UTC}. "
            "Use the supplied offline assets or obtain a fresh official URL."
        )
    print(f"Downloading official model directly from {parsed.hostname}; "
          f"URL expires {EXPIRES_UTC}", flush=True)
    command = [
        "curl", "-4", "--fail", "--location", "--proto", "=https",
        "--proto-redir", "=https", "--connect-timeout", "15",
        "--max-time", "600", "--retry", "1", "--output", str(destination),
        "--url", DIRECT_MODEL_URL,
    ]
    status = subprocess.run(command, check=False).returncode
    if status == 60:
        print("CDN certificate verification failed. Retrying this public asset "
              "with --insecure; its fixed SHA256 must still match before use.", flush=True)
        status = subprocess.run(command[:1] + ["--insecure"] + command[1:],
                                check=False).returncode
    if status:
        raise RuntimeError(
            f"Direct CDN download failed (curl exit {status}). "
            "The node may also be unable to reach the CDN; use the offline assets."
        )


def recover(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for filename, asset in ASSETS.items():
        destination = output_dir / filename
        if destination.exists() or destination.is_symlink():
            if not verified(destination, asset):
                raise ValueError(f"Existing mismatched file preserved: {destination}")
            print(f"Already verified: {destination}", flush=True)
        else:
            missing.append(filename)
    if not missing:
        return
    # An expired URL matters only when model bytes are actually needed.
    if "model.pth" in missing and time.time() >= EXPIRES_EPOCH:
        raise RuntimeError(f"This one-time CDN URL expired at {EXPIRES_UTC}; "
                           "use the supplied offline assets or obtain a fresh official URL.")
    with tempfile.TemporaryDirectory(prefix=".flow_cdn_", dir=output_dir) as temporary:
        temporary = Path(temporary)
        for filename in missing:
            staged = temporary / filename
            if filename == "model_config.yaml":
                staged.write_bytes(base64.b64decode(CONFIG_B64, validate=True))
            else:
                download_model(staged)
            if not verified(staged, ASSETS[filename]):
                raise ValueError(f"{filename}: size/SHA256 differs from the pinned official asset")
        # Verify all needed bytes before publishing either asset.
        for filename in missing:
            asset = ASSETS[filename]
            destination = output_dir / filename
            publish(temporary / filename, destination, asset)
            metadata = {
                "filename": filename, "source_repository": SOURCE_REPOSITORY,
                "model_repository": MODEL_REPOSITORY, "revision": REVISION,
                "official_url": f"{MODEL_REPOSITORY}/resolve/{REVISION}/{filename}",
                "transport": "temporary_official_cdn" if filename == "model.pth" else "embedded_original_bytes",
                "cdn_url_expires_at": EXPIRES_UTC if filename == "model.pth" else None,
                "retrieved_at": datetime.now(timezone.utc).isoformat(), **asset,
            }
            sidecar = temporary / (filename + ".source.json")
            sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
            try:
                os.link(sidecar, output_dir / sidecar.name)
            except FileExistsError:
                pass  # Preserve any earlier provenance record.
            print(f"Verified and installed: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/home/data/wyk/Flow-Planner/checkpoints")
    args = parser.parse_args()
    try:
        recover(args.output_dir)
    except Exception as error:
        raise SystemExit(f"Flow CDN recovery failed: {error}") from error
    print("Both official Flow Planner assets are verified; rerun setup_flow_eval.sh.", flush=True)


if __name__ == "__main__":
    main()
