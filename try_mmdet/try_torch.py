import torch

x = torch.Tensor([0, 1, 2])
print("w方向上的步长:", x)
y = torch.Tensor([0, 1, 2])
print("h方向上的步长:", y)
xx = x.repeat(len(y))
print(xx)
yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
print(yy)
# print(int(xx) & int(yy))
res = torch.stack([xx, yy, xx, yy], dim=-1)
print("步长网格:", res)

base_anchors = torch.Tensor([[-0.5, -0.5, .5, .5]])
print("基础锚框(插入一维后):", base_anchors[None, :, :])
print("基础锚框的size:", base_anchors[None, :, :].size())
print("网格(插入一维后):", res[:, None, :])
print("网格的size:", res[:, None, :].size())
all_anchors = base_anchors[None, :, :] + res[:, None, :]
# print("所有的anchors:", all_anchors)
# print(all_anchors.size())

all_anchors = all_anchors.view(-1, 4)
# print(all_anchors)
# print(all_anchors.size())
print("所有的anchors:", all_anchors)

#####################################################################

device = "cpu"
valid_x = torch.zeros(3, dtype=torch.bool, device=device)
valid_y = torch.zeros(3, dtype=torch.bool, device=device)
valid_x[:2] = 1
valid_y[:2] = 1
xx = valid_x.repeat(len(valid_y))
print(xx)
yy = valid_y.view(-1, 1).repeat(1, len(valid_x)).view(-1)
print(yy)
valid = xx & yy
print(valid)
valid = valid[:, None].expand(valid.size(0), 2).contiguous().view(-1)
print(valid)

#####################################################################

valid_flags = torch.tensor([[True, True, True, True], [True, True, True, True]], dtype=torch.bool, device="cpu")
flat_anchors = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
print(valid_flags)
inside_flags = valid_flags & (flat_anchors >= 5)
print(inside_flags.any())
print((None,) * 7)

#####################################################################
bboxes1 = torch.tensor([[5, 5, 5, 5], [2, 2, 2, 2]], device='cpu')
bboxes2 = torch.tensor([[3, 3, 3, 3], [4, 4, 4, 4]], device='cpu')
# bboxes1 = bboxes1[..., :, None, :2]
# print(bboxes1)
# print(bboxes1.size())
# bboxes2 = bboxes2[..., None, :, :2]
# print(bboxes2)
# print(bboxes2.size())
# lt = torch.max(bboxes1, bboxes2)
# print(lt)
# print(lt.size())
# print(lt.resize(4, 2))
# bboxes11 = bboxes1[:, None, :2]
# bboxes12 = bboxes1[:]
# print(bboxes11.size())
# print(bboxes12)

overlapss = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

# print(overlapss.max(dim=0))

print(overlapss)
print(overlapss.permute(1, 0).reshape(-1))
