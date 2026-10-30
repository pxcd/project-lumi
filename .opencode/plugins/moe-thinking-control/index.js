export default {id:'local.moe-thinking-control',async setup(ctx){
 await ctx.session.hook('context',event=>{
  if(event.model.providerID!=='lmstudio'||event.model.id!=='qwen3.6-35b-a3b-mlx')return;
  event.messages.push({role:'assistant',content:[{type:'text',text:'<think>\n\n</think>\n\n'}]});
 });
}};
