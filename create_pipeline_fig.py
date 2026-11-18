import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 300 DPI canvas proportioned for IEEE \textwidth by 4.5cm
fig, ax = plt.subplots(figsize=(16.0, 4.5), dpi=300)
ax.set_xlim(0, 16.0)
ax.set_ylim(0, 4.5)
ax.axis('off')

# Professional IEEE Academic Palette
c_p1_hdr    = '#0f4c81'
c_p1_bg     = '#f0f7fc'
c_p1_border = '#0f4c81'

c_dense_hdr = '#1b4332'
c_dense_bg  = '#ebfbee'
c_dense_brd = '#2d6a4f'

c_vit_hdr   = '#480ca8'
c_vit_bg    = '#f3e8ff'
c_vit_brd   = '#560bad'

c_fuse_hdr  = '#b45309'
c_fuse_bg   = '#fffbeb'
c_fuse_brd  = '#d97706'

c_out_hdr   = '#991b1b'
c_out_bg    = '#fef2f2'
c_out_brd   = '#b91c1c'

def draw_card(x, y, w, h, header_text, body_lines, header_color, bg_color, border_color, font_sz=8.6, text_indent=0.20):
    # Outer container
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04',
                                  facecolor=bg_color, edgecolor=border_color, linewidth=2.0)
    ax.add_patch(rect)
    
    # Header ribbon
    hh = 0.65
    header_rect = patches.FancyBboxPatch((x, y + h - hh), w, hh, boxstyle='round,pad=0.04',
                                         facecolor=header_color, edgecolor=border_color, linewidth=2.0)
    ax.add_patch(header_rect)
    
    # Header label
    ax.text(x + w/2, y + h - hh/2, header_text, color='white', fontsize=10.8,
            weight='bold', ha='center', va='center')
    
    # Body items
    n_lines = len(body_lines)
    avail_h = h - hh - 0.25
    line_spacing = avail_h / max(n_lines, 1)
    start_y = y + h - hh - 0.25
    for i, (line, is_bold, color) in enumerate(body_lines):
        ly = start_y - i * line_spacing
        ax.text(x + text_indent, ly, line, color=color, fontsize=font_sz,
                weight='bold' if is_bold else 'normal', ha='left', va='center')

def draw_arrow(x1, y1, x2, y2, label='', lbl_pos=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->,head_width=0.35,head_length=0.45', color='#1e293b', lw=2.2))
    if label:
        lx = (x1 + x2) / 2 if lbl_pos is None else lbl_pos[0]
        ly = (y1 + y2) / 2 if lbl_pos is None else lbl_pos[1]
        ax.text(lx, ly, label, fontsize=8.0, ha='center', va='center',
                weight='bold', color='#0f172a',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffff', edgecolor='#cbd5e1', lw=1.0))

# -----------------------------------------------------------------------------
# 1. Phase 1: Preprocessing & Contrast Enhancement (x: 0.15 to 2.80)
# -----------------------------------------------------------------------------
p1_lines = [
    ('• Raw Dermoscopy RGB (450x600)', True, '#0f4c81'),
    ('• DullRazor Black-Hat Filtering', False, '#1e293b'),
    ('  - Isolates dark hair occlusion filaments', False, '#475569'),
    ('  - Fast Marching Inpainting (Telea)', False, '#475569'),
    ('• CIE-LAB CLAHE Equalization', False, '#1e293b'),
    ('  - Contrast clip: 2.0 (L-channel)', False, '#475569'),
    ('  - Preserves clinical chrominance (A & B)', False, '#475569'),
    ('• Standardized Input: 224x224 RGB', True, '#0f4c81')
]
draw_card(0.15, 0.4, 2.65, 3.7, 'Phase 1: Preprocessing', p1_lines, c_p1_hdr, c_p1_bg, c_p1_border, font_sz=8.5, text_indent=0.16)

# Arrows from Phase 1
draw_arrow(2.80, 2.80, 3.85, 3.25, '[B, 3, 224, 224]', (3.30, 3.35))
draw_arrow(2.80, 1.70, 3.85, 1.25, '16x16 Patches', (3.30, 1.15))

# -----------------------------------------------------------------------------
# 2. Brain 1: DenseNet-201 Backbone (x: 3.85 to 6.55)
# -----------------------------------------------------------------------------
b1_lines = [
    ('• Pretrained DenseNet-201', True, '#1b4332'),
    ('• DenseBlocks 1-4 (Multiscale)', False, '#1e293b'),
    ('• Direct feature reuse across layers', False, '#475569'),
    ('• Extracts fine borders & textures', False, '#475569'),
    ('• Feature Map: [B, 1920, 7, 7]', True, '#1b4332')
]
draw_card(3.85, 2.35, 2.70, 1.75, 'Brain 1: DenseNet-201', b1_lines, c_dense_hdr, c_dense_bg, c_dense_brd, font_sz=8.6, text_indent=0.16)

# Arrow Brain 1 -> Brain 2 (CBAM)
draw_arrow(6.55, 3.22, 7.20, 3.22, '1920-ch', (6.88, 3.45))

# -----------------------------------------------------------------------------
# 3. Brain 2: CBAM Dual Attention (x: 7.20 to 9.95)
# -----------------------------------------------------------------------------
b2_lines = [
    ('• Channel Attention Module:', True, '#1b4332'),
    ('  AvgPool + MaxPool -> Shared MLP', False, '#475569'),
    ('• Spatial Attention Module:', True, '#1b4332'),
    ('  7x7 Conv on inter-channel pool', False, '#475569'),
    ('• Suppresses vignetting & artifacts', False, '#1e293b'),
    ('• Attentive Local Vector: 1,920-d', True, '#1b4332')
]
draw_card(7.20, 2.35, 2.75, 1.75, 'Brain 2: CBAM Dual Attention', b2_lines, c_dense_hdr, c_dense_bg, c_dense_brd, font_sz=8.6, text_indent=0.16)

# -----------------------------------------------------------------------------
# 4. Brain 3: ViT-B/16 Global Transformer (x: 3.85 to 9.95)
# -----------------------------------------------------------------------------
b3_lines = [
    ('• Vision Transformer (ViT-B/16 Architecture)', True, '#480ca8'),
    ('• 16x16 Linear Patch Projection (196 visual tokens)', False, '#1e293b'),
    ('• 12 Transformer Encoders (Multi-Head Self-Attention)', False, '#475569'),
    ('• Captures non-local lesion asymmetry & global topology', False, '#1e293b'),
    ('• Global Context Representation: [CLS] Token (768-d)', True, '#480ca8')
]
draw_card(3.85, 0.40, 6.10, 1.70, 'Brain 3: ViT-B/16 Global Context Transformer', b3_lines, c_vit_hdr, c_vit_bg, c_vit_brd, font_sz=8.6, text_indent=0.18)

# Arrows from Dual Branches to Fusion
draw_arrow(9.95, 3.22, 10.60, 2.65, '1920-d Local', (10.35, 3.05))
draw_arrow(9.95, 1.25, 10.60, 1.95, '768-d Global', (10.35, 1.50))

# -----------------------------------------------------------------------------
# 5. Phase 4: Multi-Modal Latent Fusion (x: 10.60 to 12.90)
# -----------------------------------------------------------------------------
fuse_lines = [
    ('• Feature Concatenation:', True, '#b45309'),
    ('  1,920-d (Local CBAM)', False, '#1e293b'),
    ('  + 768-d (Global ViT)', False, '#1e293b'),
    ('• Fused Latent: 2,688-d', True, '#b45309'),
    ('• Dense Linear Projection', False, '#475569'),
    ('• GELU Activation', False, '#475569'),
    ('• Dropout (p = 0.30)', False, '#475569'),
    ('• Calibrated Latent Space', True, '#b45309')
]
draw_card(10.60, 0.4, 2.30, 3.7, 'Multi-Modal Fusion', fuse_lines, c_fuse_hdr, c_fuse_bg, c_fuse_brd, font_sz=8.6, text_indent=0.16)

# Arrow Fusion -> Clinical Inference & XAI
draw_arrow(12.90, 2.25, 13.55, 2.25, '2688-d', (13.22, 2.50))

# -----------------------------------------------------------------------------
# 6. Phase 5: Safe Clinical Inference & Explainability (x: 13.55 to 15.90)
# -----------------------------------------------------------------------------
out_lines = [
    ('• Class-Weighted Focal Loss', True, '#991b1b'),
    ('  gamma = 2.0, combats class skew', False, '#475569'),
    ('• 5-Fold Spatial TTA:', True, '#991b1b'),
    ('  97.26% HAM10000 Test Acc', True, '#1e293b'),
    ('  Macro-F1: 0.9615 (N=1,497)', True, '#1e293b'),
    ('• MC Dropout Epistemic Uncertainty:', True, '#991b1b'),
    ('  15 Passes (Predictive Variance)', False, '#475569'),
    ('• DenseBlock 4 Grad-CAM XAI:', True, '#991b1b'),
    ('  Real-time margin saliency mapping', False, '#475569'),
    ('• 7-Class Calibrated Probabilities', True, '#991b1b')
]
draw_card(13.55, 0.4, 2.35, 3.7, 'Clinical Inference & XAI', out_lines, c_out_hdr, c_out_bg, c_out_brd, font_sz=8.3, text_indent=0.25)

plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)

save_p1 = r'Y:\Skin Lesion Project\pipeline_architecture.png'
save_p2 = r'Y:\Skin Lesion Project\Skin Lesion Project\pipeline_architecture.png'
plt.savefig(save_p1, dpi=300, bbox_inches='tight')
plt.savefig(save_p2, dpi=300, bbox_inches='tight')
plt.close()
print('[+] Successfully generated pixel-perfect pipeline_architecture.png!')
