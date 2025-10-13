参见脚本说明，透明画布 + 300DPI；不改变格式（按 layout_config.json 的 format 输出）。

{
  "root_dir": "./images",
  "qty_table": "./updated_products.xlsx",
  "dpi": 300,
  "canvas_w_cm": 58,
  "margin_mm": 5,

  "gutter_mm": 20,                 // 单文件夹画布的间距（mm）
  "gutter_cm": 2,                  // 大画布间距（cm）——完全按配置

  "allow_rotate": true,
  "default_count": 1,

  "format": "TIFF",                // 大画布输出格式：TIFF|PNG
  "label_mode": "sku_x_count",
  "label_include_size_cm": true,
  "label_decimals": 1,

  "per_folder_output": false,      // 是否为每个尺寸目录生成画布
  "per_folder_format": "TIFF",     // 单目录输出格式

  "big_canvas": true,              // 是否生成合并大画布
  "big_canvas_preserve_order": false,

  "big_canvas_paging": false,      // 是否分页裁切
  "big_canvas_max_height_cm": 0,   // 单页最大高度（cm，>0 时生效）
  "page_overlap_mm": 0,            // 分页重叠（mm）

  "allow_huge_image": true         // 允许打开超大图（关闭 Pillow 像素保护）
}