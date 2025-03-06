import torch
from torch import nn
from torch.masked import masked_tensor, as_masked_tensor
import warnings

# Disable prototype warnings and such
warnings.filterwarnings(action='ignore', category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100])

# data = torch.arange(12).reshape(3, 4)
# mask = data % 2 == 0
# print(mask)

# params = (masked_tensor(data, mask, requires_grad=True))

# print(params)

mask = x < 0
mask = torch.zeros(x.shape[0], dtype=torch.bool)
mx = masked_tensor(x, mask, requires_grad=True)
my = masked_tensor(torch.ones_like(x), ~mask, requires_grad=True)

print(mx, my)


p = nn.Parameter(mx, requires_grad=True)
print("params", p)


num_nys_centers = 2
input_length = 3

koopman_tensor = torch.tensor(
    torch.empty(
        (num_nys_centers*input_length, num_nys_centers*input_length),
        device=device,
        dtype=torch.float32,
    )
)

torch.nn.init.xavier_uniform_(
    koopman_tensor
)  # or any other init method

print(koopman_tensor)

mask = torch.ones(koopman_tensor.shape, dtype=torch.bool, device=device)
print(mask)
for i in range(input_length):
    for j in range(i+1, input_length):
        masked_block = torch.zeros((num_nys_centers, num_nys_centers), dtype=torch.bool)
        mask[
            num_nys_centers * i : num_nys_centers * (i + 1),
            num_nys_centers * j : num_nys_centers * (j + 1),
        ] = masked_block

print(mask)

masked_koopman_tensor = masked_tensor(koopman_tensor, mask, requires_grad=True)
print(masked_koopman_tensor)

koopman_operator = nn.Parameter(
    masked_koopman_tensor
)

print(koopman_operator)

