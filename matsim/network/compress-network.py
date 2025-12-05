import gzip  
import shutil  

src = "network.xml"         
dst = "network.xml.gz"     

with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:  
   
    shutil.copyfileobj(f_in, f_out)  

print(f"Compressed file generated: {dst}")