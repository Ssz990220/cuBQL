#!/usr/bin/python3
import os
import subprocess

num_queries=1000000
dims_list=[2,3,4,8]

#all_num_points = [ 10000, 1000000 ]
#all_num_points = [ 1000000 ]
#all_num_dims = [ 2, 3, 4 ]
#all_num_dims = [ 2 ]
#all_num_dims = [ 3 ]
point_generators = [
    [ "uniform", "uniform"],
    [ "clustered", "clustered"],
    # hot-spot/teapot:
    [ "single hot-spot", "mixture .9 remap [ .1 .2 ]  [ .15 .25 ] uniform  uniform " ],
    # cluster-outliers:
#    [ "clustered-outliers", "mixture 10 remap [ 3 4 ]  [ 5 5 ] clustered clustered"],
    # uniform-outliers:
#    [ "uniform-outliers", "mixture 10 remap [ 3 4 ]  [ 5 5 ] uniform uniform"],
    # 
    [ "shrunk uniform", "remap [ .25 ]  [ .75 ] uniform"],
    [ "shrunk clustered", "remap [ .25 ]  [ .75 ] clustered"],
    [ "hyperplane uniform", "remap [ 0 0 .1 .1 .1 .1 .1 .1 .1 ]  [ 1 1 .1 .1 .1 .1 .1 .15 .15 .15 ] uniform"],
#    [ "out-of-bounds (uniform)", "mixture .95 remap [ .25 ]  [ .75 ] uniform uniform"],
#    [ "out-of-bounds (clustered)", "mixture .95 remap [ .25 ]  [ .75 ] clustered uniform"],
    ]

def print_header(k):
    # add the multicol floatN's
    #line =" & \\multicolumn{12}{|c}{\\texttt{"
    line ="\\multicolumn{17}{|c|}{\\texttt{"
    if k == 1:
        line += "fcp} (find closest point)"
    else:
        line += "knn"+str(k)+"} (k-nearest neighbor, w/ k="+str(k)
    line += "}"
    print("\\hline")
    print(line)
    print("\\\\")
    print("\\hline")
    
    dims_line=""
    for dim in dims_list :
        dims_line += " & \multicolumn{4}{"
        #if dim == dims_list[0]:
        #    dims_line += "||"
        #else:
        dims_line += "|"
        dims_line += "c}{\\texttt{float"
        dims_line += str(dim)
        dims_line += "}}"
    print(dims_line + "\\\\")
    #
    for dim in dims_list:
        print(" & \\multicolumn{2}{|c}{N=10K} & \\multicolumn{2}{c}{N=1M}")
    print("\\\\")

def measure(generator, dims, k, count) :
    #return dims*100+k, 1000+dims*100+k, 2000+dims*100+k, 3000+dims*100+k
    # first, run bvh and generate the data_points and query_points files
    if k==1:
        k_arg = ""
    else:
        k_arg = " -k {numk}".format(numk=k)
    
    bvh_base_cmd="./cuBQL_fcpAndKnnPoints -nd {ndims} -dc {npts} -dg '{gen}' {kk} -qc {qc} -qg uniform --dump-test-data".format(ndims=dims,npts=count,gen=generator,kk=k_arg,qc=num_queries)

    bvh_cmd = bvh_base_cmd + " -lt 8"
    bvh1_cmd = bvh_base_cmd + " -lt 1"
        
    if k==1:
        ref_cmd="./cukd_float"+str(dims)+"-fcp-default"
        cct_cmd="./cukd_float"+str(dims)+"-fcp-cct"
        refxd_cmd="./cukd_float"+str(dims)+"-fcp-default-xd"
        cctxd_cmd="./cukd_float"+str(dims)+"-fcp-cct-xd"
    else:
        ref_cmd="./cukd_float"+str(dims)+"-knn-default -k "+str(k)
        cct_cmd="./cukd_float"+str(dims)+"-knn-cct -k "+str(k)
        refxd_cmd="./cukd_float"+str(dims)+"-knn-default-xd -k "+str(k)
        cctxd_cmd="./cukd_float"+str(dims)+"-knn-cct-xd -k "+str(k)
    #print("ref "+ref_cmd)
    #print("bvh "+bvh_cmd)
    #print("bvh1 "+bvh1_cmd)
    #print("cct "+cct_cmd)
    
    os.system(bvh_cmd+" > bvh.out")
    os.system(bvh1_cmd+" > bvh1.out")
    os.system(ref_cmd+" > ref.out")
    os.system(cct_cmd+" > cct.out")
    os.system(refxd_cmd+" > refxd.out")
    os.system(cctxd_cmd+" > cctxd.out")

    result = subprocess.run(['fgrep', 'STATS_DIGEST', 'bvh.out'],capture_output=True).stdout.decode('utf-8')
    print("% result bvh: >> "+result)
    #print("#bvh result : "+result)
    bvh = int(result.split()[1]) / float(num_queries)
    #print("#bvh result val "+str(bvh))

    result = subprocess.run(['fgrep', 'STATS_DIGEST', 'bvh1.out'],capture_output=True).stdout.decode('utf-8')
    print("% result bvh1: >> "+result)
    #print("#bvh result : "+result)
    bvh1 = int(result.split()[1]) / float(num_queries)
    #print("#bvh result val "+str(bvh))
    
    result = subprocess.run(['fgrep', 'KDTREE_STATS', 'ref.out'],capture_output=True).stdout.decode('utf-8')
    #print("#bvh result : "+result)
    print("% result ref: >> "+result)
    ref = int(result.split()[1]) / float(num_queries)

    result = subprocess.run(['fgrep', 'KDTREE_STATS', 'refxd.out'],capture_output=True).stdout.decode('utf-8')
    #print("#bvh result : "+result)
    print("% result refxd: >> "+result)
    refxd = int(result.split()[1]) / float(num_queries)

    result = subprocess.run(['fgrep', 'KDTREE_STATS', 'cct.out'],capture_output=True).stdout.decode('utf-8')
    #print("#bvh result : "+result)
    print("% result cct: >> "+result)
    cct = int(result.split()[1]) / float(num_queries)

    result = subprocess.run(['fgrep', 'KDTREE_STATS', 'cctxd.out'],capture_output=True).stdout.decode('utf-8')
    #print("#bvh result : "+result)
    print("% result cctxd: >> "+result)
    cctxd = int(result.split()[1]) / float(num_queries)

    #print("ref "+str(ref))
    #print("bvh "+str(bvh))
    #print("cct "+str(cct))
    return ref, bvh, bvh1, cct, refxd, cctxd

def three_digits_number(n):
    if n >= 100 :
        return str(int(n))
    if (n % 1.) < .05:
        return "{:3.0f}".format(round(n))
    return "{:.1f}".format(n)
    
def pretty_number(n) :
    if n < 1000 :
        return three_digits_number(n)
    if n < 1000000 :
        return three_digits_number(n/1000.)+"~K"
    if n < 1000000000 :
        return three_digits_number(n/1000000.)+"~M"
    if n < 1000000000000 :
        return three_digits_number(n/1000000000.)+"~G"
    return "<cannot format "+str(n)+">"

def pretty_relAccesses(s) :
    if s < .8 :
        pct = int(100*(1-s))
        return "\\bigWinPct{-"+str(pct)+"}"
    elif s <= 1. :
        pct = int(100*(1-s))
        return "\\lowWinPct{-"+str(pct)+"}"
    else:
        pct = int(100*(s-1))
        return "\\lossPct{+"+str(pct)+"}"
    
def pretty_speedup(s) :
    number = ""
    if s < 1. :
        pct = -(s-1.)*100;
        number=str(int(pct))
        return "\\lossPct{"+number+"}"
    elif s < 1.2 :
        pct = (s-1.)*100;
        number=str(int(pct))
        return "\\winPct{"+number+"}"
    elif s < 10 :
        number="{:.1f}".format(s)
        return "\\winFactor{"+number+"}"
    else :
        number=str(int(s))
        return "\\winFactor{"+number+"}"

def do_distribution(tex_name, generator_string, k):
    print("""
    \hline
    \hline
    \multicolumn{17}{c}{"""+tex_name+"\\quad\\quad{\\relsize{-2}{\\texttt{('"+generator_string+"')}}}"+"""} \\\\
    \hline""")
    ref_line = "kd-tree (default)"
    refxd_line = "kd-tree (xd)"
    #bvh_line = "bvh(lt=k/2) "
    bvh_line  = "bvh(lt=8) "
    bvh1_line = "bvh(lt=1) "
    cct_line = "kd-tree (cdpt) "
    cctxd_line = "kd-tree (cdpt,xd) "
    for dims in dims_list :
            for count in [ 10000, 1000000 ] :
                ref, bvh, bvh1, cct, refxd, cctxd = measure(generator_string, dims, k, count)
                #print(" ---- ")
                #print("ref "+str(ref))
                #print("bvh "+str(bvh))
                #print("cct "+str(cct))
                best = min(min(min(ref,cct),min(bvh,bvh1)),min(refxd,cctxd))
                
                #if k==1:
                #    bvh_cell = "(n/a)"
                #    bvh_speedup = ""
                #else:
                bvh_cell = pretty_number(bvh)
                bvh_speedup = pretty_relAccesses(bvh*1./ref)
                if bvh == best :
                    bvh_cell = "\\textbf{"+bvh_cell+"}"

                bvh1_cell = pretty_number(bvh1)
                if bvh1 == best :
                    bvh1_cell = "\\textbf{"+bvh1_cell+"}"
                bvh1_speedup = pretty_relAccesses(bvh1*1./ref)

                cct_cell = pretty_number(cct)
                cct_speedup = pretty_relAccesses(cct*1./ref)
                if cct == best :
                    cct_cell = "\\textbf{"+cct_cell+"}"

                cctxd_cell = pretty_number(cctxd)
                cctxd_speedup = pretty_relAccesses(cctxd*1./ref)
                if cctxd == best :
                    cctxd_cell = "\\textbf{"+cctxd_cell+"}"

                ref_cell = pretty_number(ref)
                ref_speedup = ""
                if ref == best :
                    ref_cell = "\\textbf{"+ref_cell+"}"

                refxd_cell = pretty_number(refxd)
                refxd_speedup = ""
                if refxd == best :
                    refxd_cell = "\\textbf{"+refxd_cell+"}"

                ref_line = ref_line + "& " + ref_cell + " & " + ref_speedup
                refxd_line = refxd_line + "& " + refxd_cell + " & " + refxd_speedup
                cct_line = cct_line + "& " + cct_cell + " & " + cct_speedup
                cctxd_line = cctxd_line + "& " + cctxd_cell + " & " + cctxd_speedup
                bvh_line = bvh_line + "& " + bvh_cell + " & " + bvh_speedup
                bvh1_line = bvh1_line + "& " + bvh1_cell + " & " + bvh1_speedup
    print(ref_line)
    print("\\\\")
    print(refxd_line)
    print("\\\\")
    print(cct_line)
    print("\\\\")
    print(cctxd_line)
    print("\\\\")
    print(bvh1_line)
    print("\\\\")
    print(bvh_line)
    print("\\\\")

def make_table(k):
    # make "tabular" line
    print("\\begin{table*}")
    print("\\setlength{\\tabcolsep}{3pt}")
    tab_line="{\\relsize{-1}{\\begin{tabular}{l"
#    for dim in dims_list:
    tab_line += "||cccc|cccc|cccc|cccc"
    tab_line += "}"
    print(tab_line)
    #
    #
    print_header(k)
    for gen in point_generators :
        do_distribution(gen[0],gen[1],k)
    print("\\end{tabular}}}")
    print("\\caption{table for fctAndKnn("+str(k)+")}")
    print("\\end{table*}")
    
def main():
    for k in [ 1, 8, 64 ] :
        make_table(k)

main()
