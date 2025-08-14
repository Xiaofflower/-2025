import cv2
import numpy as np
import os

def embed_watermark(original_image, watermark_image, alpha=0.1):
    print("开始嵌入水印...")
    if original_image.shape[:2] != watermark_image.shape[:2]:
        watermark_image = cv2.resize(watermark_image, (original_image.shape[1], original_image.shape[0]))


    b, g, r = cv2.split(original_image)

  
 
    # 公式: dst = src1*alpha + src2*beta + gamma
    embedded_b = cv2.addWeighted(b, 1.0, watermark_image, alpha, 0)

   
    watermarked_img = cv2.merge((embedded_b, g, r))
    print("水印嵌入完成。")
    return watermarked_img


def extract_watermark(watermarked_image, original_image):
   
   
    b_watermarked, _, _ = cv2.split(watermarked_image)
    b_original, _, _ = cv2.split(original_image)

   
    extracted_watermark = cv2.absdiff(b_watermarked, b_original)
    
   
    extracted_watermark = cv2.normalize(extracted_watermark, None, 0, 255, cv2.NORM_MINMAX)

    return extracted_watermark


if __name__ == "__main__":
  
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

   
    print("读取原始文件...")
    original_img = cv2.imread(ORIGINAL_IMG_PATH)
    if original_img is None:
        raise FileNotFoundError(f"错误：无法找到或读取原始图像 '{ORIGINAL_IMG_PATH}'")
    
    watermark_img = cv2.imread(WATERMARK_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if watermark_img is None:
        raise FileNotFoundError(f"错误：无法找到或读取水印图像 '{WATERMARK_IMG_PATH}'")

   
    watermarked_img = embed_watermark(original_img.copy(), watermark_img.copy())
    watermarked_path = os.path.join(OUTPUT_DIR, "watermarked.png")
    cv2.imwrite(watermarked_path, watermarked_img)
    print(f"带水印的图片已保存到: {watermarked_path}")

   
    print("\n--- 无攻击情况下的水印提取 ---")
    extracted_clean = extract_watermark(watermarked_img, original_img)
    extracted_clean_path = os.path.join(OUTPUT_DIR, "extracted_from_clean.png")
    cv2.imwrite(extracted_clean_path, extracted_clean)
    print(f"成功提取，结果保存到: {extracted_clean_path}")


  
    print("\n--- 开始鲁棒性测试 ---")

  
    print("测试A: 翻转攻击...")
    attacked_flipped = cv2.flip(watermarked_img, 1)
    original_flipped = cv2.flip(original_img, 1) # 原始图也需同样变换
    extracted_from_flip = extract_watermark(attacked_flipped, original_flipped)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "attacked_flip.png"), attacked_flipped)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "extracted_from_flip.png"), extracted_from_flip)
    print("翻转攻击测试完成。")

   
    print("测试B: 平移攻击...")
    rows, cols, _ = watermarked_img.shape
    M = np.float32([[1, 0, 50], [0, 1, 30]]) # 向右平移50，向下平移30
    attacked_shifted = cv2.warpAffine(watermarked_img, M, (cols, rows))
    original_shifted = cv2.warpAffine(original_img, M, (cols, rows)) # 原始图也需同样变换
    extracted_from_shift = extract_watermark(attacked_shifted, original_shifted)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "attacked_shift.png"), attacked_shifted)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "extracted_from_shift.png"), extracted_from_shift)
    print("平移攻击测试完成。")

  
    print("测试C: 对比度攻击...")
    attacked_contrast = cv2.convertScaleAbs(watermarked_img, alpha=1.5, beta=10)
    original_contrast = cv2.convertScaleAbs(original_img, alpha=1.5, beta=10) # 原始图也需同样变换
    extracted_from_contrast = extract_watermark(attacked_contrast, original_contrast)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "attacked_contrast.png"), attacked_contrast)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "extracted_from_contrast.png"), extracted_from_contrast)
    print("对比度攻击测试完成。")

    
    print("测试D: 截取攻击...")
   
    attacked_cropped = watermarked_img[100:rows-100, 100:cols-100]
    original_cropped = original_img[100:rows-100, 100:cols-100] # 原始图也需同样变换
    extracted_from_crop = extract_watermark(attacked_cropped, original_cropped)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "attacked_crop.png"), attacked_cropped)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "extracted_from_crop.png"), extracted_from_crop)
    print("截取攻击测试完成。")

    print("\n所有测试已完成，请查看 'output' 文件夹中的结果。")
