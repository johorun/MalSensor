from urllib.request import CacheFTPHandler
import idaapi
import idautils
import idc
import time
# def function_extract(output_file, func, callees):
#     func_name = get_func_name(func)
#     print ("Function Name:%s" % (func_name) , file = output_file)
#     for ref_ea in CodeRefsTo(func, 1):    
#         caller_name = get_func_name(ref_ea)
#         callees[caller_name] = callees.get(caller_name, set()) #add the functions from "CodesRefsTo" to a dictionary for extracting CG and CG adjacency Matrix
#         callees[caller_name].add(func_name)  
#         print ( "		%s" % (caller_name), file = output_file ) 

def callback(ea, name, ordinal):
    imports[current].append((ea, name, ordinal))
    return True    

def controller():
    imp_API = []
    imports = {}
    current = ""
    def callback(ea, name, ordinal):
        imports[current].append((ea, name, ordinal))
        return True 
    #print(ida_nalt.get_import_module_qty()) 函数库的数量索引
    FileType = idaapi.get_file_type_name() #获取文件类型
    basename = ida_nalt.get_root_filename()
    if FileType == 'Portable executable for 80386 (PE)' or FileType == 'Portable executable for AMD64 (PE)':
        info_filename = "I:\\20220728\MalwareBazaar\\PE_packed\\"+ basename + ".info"
    else:
        info_filename = "I:\\20220728\MalwareBazaar\\noPE_packed\\"+ basename + ".info"
    
    callees = dict()        
    tic = time.time()
    #asm_filename = basename + ".asm"     
    #idc.gen_file(idc.OFILE_ASM, basename + ".asm", 0, idc.BADADDR, 0)       
    #output_file = open(info_filename,'w')        
    funcs = idautils.Functions() #funcs is entrypoint返回的是入口点也就是地址     
    for f in funcs:        
        #function_extract(output_file, f, callees) # extract functions data
        func_name = get_func_name(f)
        for caller in CodeRefsTo(f, 1):
            caller_name = get_func_name(caller)
            callees.setdefault(func_name,[]).append(caller_name)
    
    nimps = ida_nalt.get_import_module_qty()
    for i in range(0, nimps):
        current = ida_nalt.get_import_module_name(i)
        imports[current] = []
        ida_nalt.enum_import_names(i, callback)
    for key in imports:
        for funcs in imports[key]:
            func_name = funcs[1]        #(4350228, 'WaitForSingleObjectEx', 0)
            imp_API.append(func_name)
            f = funcs[0]
            for caller in CodeRefsTo(f, 1):
                caller_name = get_func_name(caller)
                callees.setdefault(func_name,[]).append(caller_name)
#idautils先把整个图构建起来，再用imports去把所有的外部导入函数有遗漏的补全
                        
    tic = time.time() - tic
    with open('I:\\20220728\\new.txt', 'a+') as f:
        f.write(basename+'  ')
        f.write(str(tic)+'\n')
    with open(info_filename,'w') as f:
        for func_name in callees:
            tmp = func_name
            demangled = idc.demangle_name(func_name, idc.get_inf_attr(idc.INF_SHORT_DN))
            if demangled is not None:
                func_name = demangled
            f.write("Function Name:" + func_name + '\n')
            for caller_name in callees[tmp]:
                demangled = idc.demangle_name(caller_name, idc.get_inf_attr(idc.INF_SHORT_DN))
                if demangled is not None:
                    caller_name = demangled
                f.write(caller_name + '\n')
    with open('I:\\20220728\\imp_API.txt', 'a+') as f:
        for API in imp_API:
            demangled = idc.demangle_name(API, idc.get_inf_attr(idc.INF_SHORT_DN))
            if demangled is not None:
                API = demangled
            f.write(API)
            f.write('\n')     
   
# end of controller
q = None
f = None
ida_auto.auto_wait()
controller()
ida_pro.qexit(0)

