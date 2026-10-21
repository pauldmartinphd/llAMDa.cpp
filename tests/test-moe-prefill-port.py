from pathlib import Path
import subprocess, sys
r=Path(sys.argv[1]).resolve(); build=Path(sys.argv[2]).resolve()
s=(r/'ggml/src/ggml-backend.cpp').read_text(); a=s.index('                    // Keep each layer'); b=s.index('\n                }\n                SET_CAUSE',a); selection=s[a:b]
s=(r/'src/llama-context.cpp').read_text(); a=s.index('bool llama_context::sched_copy_experts('); b=s.index('\nllm_graph_cb llama_context::graph_get_cb()',a); copy=s[a:b]
code=r'''
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdint>
#define GGML_SCHED_MAX_BACKENDS 16
#define SET_CAUSE(...)
struct mock_sched { bool moe_prefill = true; unsigned off_rr = 0; int backends[5] = {0,1,2,3,4}; };
static bool eligible_devices[4] = {true,true,true,true};
static bool supports(int b, ggml_tensor *) {return eligible_devices[b];}
#define ggml_backend_supports_op supports
#define ggml_backend_offload_op supports
int select_backend(mock_sched * sched, ggml_tensor * tensor) {
 const auto * src = tensor->src[0]; const int src_backend_id = 4; const int i = 0;
''' + selection + r'''
 return -1;
}
#undef ggml_backend_supports_op
#undef ggml_backend_offload_op
struct mock_cache {
 bool copy(ggml_backend_t, const ggml_tensor *, ggml_tensor *, ggml_cgraph *) {return false;}
 int64_t copy_experts(ggml_backend_t, const ggml_tensor *, ggml_tensor *, int64_t, int64_t) {return 0;}
};
struct llama_context {
 mock_cache * moe_cache = nullptr;
 struct {const ggml_tensor * ids = nullptr; std::vector<int32_t> ids_data; std::vector<bool> used;} copy_experts;
 static bool sched_copy_experts(ggml_backend_t, const ggml_tensor *, ggml_tensor *, ggml_cgraph *, void *);
};
static int reads = 0, syncs = 0;
static void read_ids(ggml_backend_t b, const ggml_tensor * t, void * p, size_t o, size_t n) {++reads; ggml_backend_tensor_get_async(b,t,p,o,n);}
static void sync_ids(ggml_backend_t b) {++syncs; ggml_backend_synchronize(b);}
#define ggml_backend_tensor_get_async read_ids
#define ggml_backend_synchronize sync_ids
''' + copy + r'''
#undef ggml_backend_tensor_get_async
#undef ggml_backend_synchronize
void check_copy(int nt, bool enabled) {
 if (enabled) unsetenv("GGML_MOE_PREFILL"); else setenv("GGML_MOE_PREFILL","0",1);
 ggml_init_params ip = {4*1024*1024,nullptr,true}; auto * ctx=ggml_init(ip);
 auto * src=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,4,4,8);
 auto * dst=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,4,4,8);
 auto * x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,4,1,nt);
 auto * ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,2,nt);
 auto * out=ggml_mul_mat_id(ctx,dst,x,ids);
 auto * graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,out);
 auto * backend=ggml_backend_cpu_init(); auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
 for (int i=0;i<128;i++) ((float*)src->data)[i]=float(i+1);
 std::fill_n((float*)dst->data,128,-1.0f);
 // Skew all routing to expert 0; copy-all must still produce identical output.
 std::fill_n((int32_t*)ids->data,2*nt,0);
 std::fill_n((float*)x->data,4*nt,1.0f);
 llama_context lc; reads=syncs=0;
 assert(llama_context::sched_copy_experts(backend,src,dst,graph,&lc));
 const bool all=enabled && nt>=32;
 assert(reads==(all?0:1) && syncs==(all?0:1));
 assert(((float*)dst->data)[0]==1.0f);
 if(all) assert(((float*)dst->data)[127]==128.0f);
 assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
 for(int t=0;t<2*nt;t++) for(int row=0;row<4;row++) assert(((float*)out->data)[4*t+row]==float(16*row+10));
 // Reusing the same routing tensor must not read it again.
 assert(llama_context::sched_copy_experts(backend,src,dst,graph,&lc));
 assert(reads==(all?0:1));
 ggml_backend_buffer_free(buffer); ggml_backend_free(backend); ggml_free(ctx);
}
int main() {
 ggml_tensor ids{}, w{}, t{}; ids.ne[1]=32; t.op=GGML_OP_MUL_MAT_ID; t.src[0]=&w; t.src[2]=&ids;
 mock_sched sched;
 for(int layer=0;layer<16;layer++) for(const char * part:{"gate","up","down"}) {
  snprintf(w.name,sizeof(w.name),"blk.%d.ffn_%s_exps.weight",layer,part);
  assert(select_backend(&sched,&t)==layer%4);
 }
 eligible_devices[1]=eligible_devices[3]=false;
 for(int layer=0;layer<8;layer++) {snprintf(w.name,sizeof(w.name),"blk.%d.ffn_up_exps.weight",layer); assert(select_backend(&sched,&t)==(layer%2)*2);}
 strcpy(w.name,"unnamed"); assert(select_backend(&sched,&t)==0); assert(select_backend(&sched,&t)==2);
 strcpy(w.name,"blk.3.ffn_up_exps.weight"); ids.ne[1]=1; assert(select_backend(&sched,&t)==0);
 ids.ne[1]=32; sched.moe_prefill=false; assert(select_backend(&sched,&t)==0);
 sched.moe_prefill=true; t.op=GGML_OP_MUL_MAT; assert(select_backend(&sched,&t)==0);
 eligible_devices[0]=eligible_devices[2]=false; assert(select_backend(&sched,&t)==-1);
 for(int n:{1,31,32,33,128}) {check_copy(n,true);check_copy(n,false);}
 puts("PASS: four-device placement, eligible subset, gate/up/down affinity, fallback, decode/non-MoE unchanged, bypass boundary, skewed routing numerical equivalence, selective ID reuse, A/B switch");
}
'''
p=build/'test-prefill-port.cpp'; p.write_text(code)
cmd=['clang++','-std=c++17','-I'+str(r/'ggml/include'),str(p),'-L'+str(build/'bin'),'-lggml-base','-lggml-cpu','-Wl,-rpath,'+str(build/'bin'),'-o',str(build/'test-prefill-port')]
subprocess.run(cmd,check=True); subprocess.run([str(build/'test-prefill-port')],check=True)
