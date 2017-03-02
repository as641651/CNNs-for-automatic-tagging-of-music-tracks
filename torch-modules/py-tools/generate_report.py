import os

def write_tex_report(im_dir, auc_imgs, ap_imgs):
  
  code = ""
  code += "\documentclass{article}\n"
  code += "\usepackage{graphicx}\n"
  code += "\graphicspath{{%s}}\n\n"%im_dir

  code += "\\begin{document}\n\n"

  for i in range(len(auc_imgs)):
    code += "\\begin{figure}\n\n"
    code += "\centering\n"
    code += "\\begin{minipage}{.45\linewidth}\n"
    code += "  \includegraphics[width=\linewidth]{%s}\n"%auc_imgs[i]
    code += "\end{minipage}\n"
    code += "\hspace{.05\linewidth}\n"
    code += "\\begin{minipage}{.45\linewidth}\n"
    code += "  \includegraphics[width=\linewidth]{%s}\n"%ap_imgs[i]
    code += "\end{minipage}\n"
    code += "\hspace{.05\linewidth}\n"
    code += "\\end{figure}\n\n"

  code += "\\end{document}\n"

  tex_file = open(os.path.join(im_dir,"report.tex"),"w")
  tex_file.write(code)
  tex_file.close()

if __name__ == '__main__':

  imgs = []
  imgs.append("auc_adj.png")
  imgs.append("auc_genre.png")
  imgs.append("auc_pace.png")
  code = write_tex_report("imgs/",imgs)
  print(code)
