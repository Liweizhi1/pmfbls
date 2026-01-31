import torch
import torch.nn as nn
from bls_ultimate import ProtoNet_MiniBLS_Ultimate

# ==========================================
# 1. 模拟环境 (修正版)
# ==========================================
class MockViTSmall(nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟 12 层 transformer blocks
        # 使用 Identity 层，虽然不做计算，但会让数据流过去，从而触发 Hook
        self.blocks = nn.ModuleList([
            nn.Identity() for _ in range(12) 
        ])
        self.embed_dim = 384 

    def forward(self, x):
        # 【关键修正】
        # 我们得先把图片变成特征形状 [B, 197, 384]，模拟 Patch Embedding
        B = x.shape[0]
        # 随机生成一个符合 ViT-Small 维度的特征
        feat = torch.randn(B, 197, 384, device=x.device)
        
        # 【必须让数据真正流过每一层】
        # 这样 bls_ultimate.py 里的 Hook 才能抓到数据！
        for layer in self.blocks:
            feat = layer(feat)
            
        return feat

# 假装这是你的 args
class MockArgs:
    mini_bls_mlf_layers = "-1,-3,-6"
    mini_bls_mapping_dim = 1000
    mini_bls_reg_lambda = 1e-3

# ==========================================
# 2. 正式运行流程
# ==========================================
if __name__ == "__main__":
    print(">>> 1. 正在初始化 Backbone (ViT-Small)...")
    backbone = MockViTSmall()
    args = MockArgs()

    print(">>> 2. 正在加载你的究极版 BLS 模型...")
    model = ProtoNet_MiniBLS_Ultimate(backbone, args)

    # 3. 造一点假数据 (5-way 1-shot)
    x_sup = torch.randn(2, 5, 1, 3, 224, 224) 
    y_sup = torch.randint(0, 5, (2, 5))       
    x_qry = torch.randn(2, 75, 3, 224, 224)

    print(">>> 3. 开始前向推理 (Forward)...")
    if torch.cuda.is_available():
        model = model.cuda()
        x_sup, y_sup, x_qry = x_sup.cuda(), y_sup.cuda(), x_qry.cuda()
    
    with torch.no_grad():
        logits = model(x_sup, y_sup, x_qry)

    print("------------------------------------------------")
    print(f"✅ 运行成功！没有报错了！")
    print(f"输出 Logits 形状: {logits.shape}")
    print("------------------------------------------------")