# governance/ — model cards

| Path | Contents |
|------|----------|
| `model-cards/` | [`tfm_fraud_heads.md`](./model-cards/tfm_fraud_heads.md) — the TFM + XGBoost-head ensemble; [`example_model_card.md`](./model-cards/example_model_card.md) — the blank template. |
| `policies/` *(add as needed)* | Ranger/SDX policies for the Impala/VAST tables. |

## Conventions

- Secrets never live in the repo: VAST keys in the untracked `.vast.env`.
- The lift numbers shown in demos come from the live engine — refresh the card when the
  data or heads change materially.
