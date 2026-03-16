## Notas extra

### Matrices

- 0 Columnas.
- 1 Filas.

```python
x = torch.tensor([1, 2, 3, 4])
torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])´´´
```
