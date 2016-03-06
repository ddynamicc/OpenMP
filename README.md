void process_sao(Context* h265ctx, H265Picture* pic, int8 is_luma_used, int8 is_chroma_used, Tile* tile, uint8 is_tile_boundary_used)
{
  uint32 height = pic->size->height;
  uint32 width = pic->size->width;
  uint32 stride = pic->recon_yuv.stride;
  uint32 cstride = pic->recon_yuv.cstride;
  int16* src_y = pic->recon_yuv.buf_y;
  int16* src_u = pic->recon_yuv.buf_u;
  int16* src_v = pic->recon_yuv.buf_v;   

  int32 layer_num=pic->m_layerId;
 
  // [JH] potential bug exists.
  // H265ctx structure contains CTU information and pointers for main structure (H265Slice, H265Picture etc.).
  // As sample adaptive offset (SAO) is selectively used according to slice segment, initialization process of the pointers is considered to properly access current slice segment.  

#if Inloop_openMP
  int32 num_ths = omp_get_num_procs();
  int32 ctu_num;
  int32 i=0;

  for( i = 0; i < num_ths; i++)
  {
#if LAYER_PARALLEL_MEM
	  memcpy(h265ctxInloop[layer_num][i], h265ctx, sizeof(Context));
#else
	  memcpy(h265ctxInloop[i], h265ctx, sizeof(Context));
#endif
  }

  omp_set_num_threads(num_ths);
#endif
    
  if (is_luma_used)
  {
#if Inloop_openMP
	int16* dst_OpenMp[MAX_TILE_PER_PICTURE];      
	uint32 pos_y; 
	int32 stride = pic->recon_yuv.stride;
#else
	uint32 pos_y; 
	uint32 ctu_x, ctu_y;
	int32 ctu_address; 
	int32 sao_type;
	int16* dst;
	int32 stride = pic->recon_yuv.stride;
#endif
	
    // copy frame buffer
    for(pos_y = 0; pos_y < height; pos_y++)
    {
#if LAYER_PARALLEL_MEM
	  memcpy(inloop_mem_y[layer_num][pos_y], src_y, width*sizeof(int16));
#else
      memcpy(inloop_mem_y[pos_y], src_y, width*sizeof(int16));
#endif
      src_y += stride;
    }  

    // CTU-based processing
#if Inloop_openMP
#pragma omp parallel for schedule(dynamic) num_threads(num_ths)
	for (ctu_num = 0; ctu_num< pic->size->ctu_num_in_height * pic->size->ctu_num_in_width; ctu_num++)
	{ 
		int32 thread_num = omp_get_thread_num();
		uint32 ctu_x, ctu_y;
		int32 ctu_address; 
		int32 sao_type;

		ctu_address = ctu_num;
		ctu_x = ctu_address % pic->size->ctu_num_in_width;
		ctu_y = ctu_address / pic->size->ctu_num_in_width;

#if LAYER_PARALLEL_MEM
		sao_type = sao_params[layer_num][COMP_Y][ctu_address].category;
#else
		sao_type = sao_params[COMP_Y][ctu_address].category;
#endif
		if (sao_type > 0)
		{
#if LAYER_PARALLEL_MEM
			init_ctu_context_for_sao(h265ctxInloop[layer_num][thread_num], pic, ctu_address, ctu_y, ctu_x, h265ctxInloop[layer_num][thread_num]->slice->pps->is_loop_filter_used_across_tile, h265ctxInloop[layer_num][thread_num]->slice->is_loop_filter_cross_slice);
			dst_OpenMp[thread_num] = pic->recon_yuv.buf_y + get_ctu_luma_offset(h265ctxInloop[layer_num][thread_num], 0);

			process_sao_type[sao_type - 1][COMP_Y](h265ctxInloop[layer_num][thread_num], ctu_y, ctu_x, ctu_address, dst_OpenMp[thread_num], stride, pic->size, tile, is_tile_boundary_used);                                      

			if(h265ctxInloop[layer_num][thread_num]->slice->sps->is_pcm_used && h265ctxInloop[layer_num][thread_num]->slice->sps->is_pcm_filtering_used || h265ctxInloop[layer_num][thread_num]->slice->pps->is_bypass_used)
			{
				restore_pcm_luma(h265ctxInloop[layer_num][thread_num], ctu_address, ctu_y, ctu_x);
			}
#else
			init_ctu_context_for_sao(h265ctxInloop[thread_num], pic, ctu_address, ctu_y, ctu_x, h265ctxInloop[thread_num]->slice->pps->is_loop_filter_used_across_tile, h265ctxInloop[thread_num]->slice->is_loop_filter_cross_slice);
			dst_OpenMp[thread_num] = pic->recon_yuv.buf_y + get_ctu_luma_offset(h265ctxInloop[thread_num], 0);

			process_sao_type[sao_type - 1][COMP_Y](h265ctxInloop[thread_num], ctu_y, ctu_x, ctu_address, dst_OpenMp[thread_num], stride, pic->size, tile, is_tile_boundary_used);                                      

			if(h265ctxInloop[thread_num]->slice->sps->is_pcm_used && h265ctxInloop[thread_num]->slice->sps->is_pcm_filtering_used || h265ctxInloop[thread_num]->slice->pps->is_bypass_used)
			{
				restore_pcm_luma(h265ctxInloop[thread_num], ctu_address, ctu_y, ctu_x);
			}
#endif
		}

	}

#else
	for (ctu_y = 0; ctu_y< pic->size->ctu_num_in_height; ctu_y++)
    { 
      for (ctu_x = 0; ctu_x < pic->size->ctu_num_in_width; ctu_x++)
      {
        ctu_address = ctu_y * pic->size->ctu_num_in_width + ctu_x;
#if LAYER_PARALLEL_MEM
		sao_type = sao_params[layer_num][COMP_Y][ctu_address].category;
#else
        sao_type = sao_params[COMP_Y][ctu_address].category;
#endif

        if (sao_type > 0)
        {
          init_ctu_context_for_sao(h265ctx, pic, ctu_address, ctu_y, ctu_x, h265ctx->slice->pps->is_loop_filter_used_across_tile, h265ctx->slice->is_loop_filter_cross_slice);
          dst = pic->recon_yuv.buf_y + get_ctu_luma_offset(h265ctx, 0);
  
          process_sao_type[sao_type - 1][COMP_Y](h265ctx, ctu_y, ctu_x, ctu_address, dst, stride, pic->size, tile, is_tile_boundary_used);                                      

          if(h265ctx->slice->sps->is_pcm_used && h265ctx->slice->sps->is_pcm_filtering_used || h265ctx->slice->pps->is_bypass_used)
          {
            restore_pcm_luma(h265ctx, ctu_address, ctu_y, ctu_x);
          }
        }
      }
    }
#endif
  }
  
  // SAO for chroma
  if (is_chroma_used)
  {
#if Inloop_openMP
	int16* dst_OpenMp_chroma[MAX_TILE_PER_PICTURE]; 
	uint32 pos_y;  //, ctu_x, ctu_y, ctu_address, sao_type;
	int32 stride = pic->recon_yuv.cstride;
#else
	uint32 pos_y;  //, ctu_x, ctu_y, ctu_address, sao_type;
	int32 stride = pic->recon_yuv.cstride;
	uint32 ctu_x, ctu_y;
	int32 ctu_address; 
	int32 sao_type;
	int16* dst;
#endif

    for(pos_y = 0; pos_y < (height >> 1); pos_y++)
    {
#if LAYER_PARALLEL_MEM
	  memcpy(inloop_mem_u[layer_num][pos_y], src_u, (width >> 1)*sizeof(int16));
	  memcpy(inloop_mem_v[layer_num][pos_y], src_v, (width >> 1)*sizeof(int16));
#else
      memcpy(inloop_mem_u[pos_y], src_u, (width >> 1)*sizeof(int16));
      memcpy(inloop_mem_v[pos_y], src_v, (width >> 1)*sizeof(int16));
#endif
      src_u += cstride;
      src_v += cstride;
    }   

#if Inloop_openMP
#pragma omp parallel
	{
		int32 thread_num = omp_get_thread_num();
		uint32 ctu_x, ctu_y;
		int32 ctu_address; 
		int32 sao_type;

#pragma omp for schedule(dynamic) //num_threads(num_ths)
		for (ctu_num = 0; ctu_num< pic->size->ctu_num_in_height * pic->size->ctu_num_in_width; ctu_num++)
		{ 
			ctu_address = ctu_num;
			ctu_x = ctu_address % pic->size->ctu_num_in_width;
			ctu_y = ctu_address / pic->size->ctu_num_in_width;
#if LAYER_PARALLEL_MEM
			sao_type = sao_params[layer_num][COMP_U][ctu_address].category;
#else
			sao_type = sao_params[COMP_U][ctu_address].category;                
#endif

#if LAYER_PARALLEL_MEM
			init_ctu_context_for_sao(h265ctxInloop[layer_num][thread_num], pic, ctu_address, ctu_y, ctu_x, h265ctxInloop[layer_num][thread_num]->slice->pps->is_loop_filter_used_across_tile, h265ctxInloop[layer_num][thread_num]->slice->is_loop_filter_cross_slice);
			if (sao_type > 0)
			{                   
				dst_OpenMp_chroma[thread_num] = h265ctxInloop[layer_num][thread_num]->pic->recon_yuv.buf_u + get_ctu_chroma_offset(h265ctxInloop[layer_num][thread_num], 0);
				process_sao_type[sao_type - 1][COMP_U](h265ctxInloop[layer_num][thread_num], ctu_y, ctu_x, ctu_address, dst_OpenMp_chroma[thread_num], stride, pic->size, tile, is_tile_boundary_used);
			}   
#if LAYER_PARALLEL_MEM
			sao_type = sao_params[layer_num][COMP_U][ctu_address].category;
#else
			sao_type = sao_params[COMP_U][ctu_address].category;
#endif
			if (sao_type > 0)
			{          
				dst_OpenMp_chroma[thread_num] = h265ctxInloop[layer_num][thread_num]->pic->recon_yuv.buf_v + get_ctu_chroma_offset(h265ctxInloop[layer_num][thread_num], 0);
				process_sao_type[sao_type - 1][COMP_V](h265ctxInloop[layer_num][thread_num], ctu_y, ctu_x, ctu_address, dst_OpenMp_chroma[thread_num], stride, pic->size, tile, is_tile_boundary_used);
			}   

			if(h265ctxInloop[layer_num][thread_num]->slice->sps->is_pcm_used && h265ctxInloop[layer_num][thread_num]->slice->sps->is_pcm_filtering_used || h265ctxInloop[layer_num][thread_num]->slice->pps->is_bypass_used)
			{
				restore_pcm_chroma(h265ctxInloop[layer_num][thread_num], ctu_address, ctu_y, ctu_x);
			}
#else
			init_ctu_context_for_sao(h265ctxInloop[thread_num], pic, ctu_address, ctu_y, ctu_x, h265ctxInloop[thread_num]->slice->pps->is_loop_filter_used_across_tile, h265ctxInloop[thread_num]->slice->is_loop_filter_cross_slice);
			if (sao_type > 0)
			{                   
				dst_OpenMp_chroma[thread_num] = h265ctxInloop[thread_num]->pic->recon_yuv.buf_u + get_ctu_chroma_offset(h265ctxInloop[thread_num], 0);
				process_sao_type[sao_type - 1][COMP_U](h265ctxInloop[thread_num], ctu_y, ctu_x, ctu_address, dst_OpenMp_chroma[thread_num], stride, pic->size, tile, is_tile_boundary_used);
			}   

			sao_type = sao_params[COMP_U][ctu_address].category;
			if (sao_type > 0)
			{          
				dst_OpenMp_chroma[thread_num] = h265ctxInloop[thread_num]->pic->recon_yuv.buf_v + get_ctu_chroma_offset(h265ctxInloop[thread_num], 0);
				process_sao_type[sao_type - 1][COMP_V](h265ctxInloop[thread_num], ctu_y, ctu_x, ctu_address, dst_OpenMp_chroma[thread_num], stride, pic->size, tile, is_tile_boundary_used);
			}   

			if(h265ctxInloop[thread_num]->slice->sps->is_pcm_used && h265ctxInloop[thread_num]->slice->sps->is_pcm_filtering_used || h265ctxInloop[thread_num]->slice->pps->is_bypass_used)
			{
				restore_pcm_chroma(h265ctxInloop[thread_num], ctu_address, ctu_y, ctu_x);
			}
#endif
		}
	}
#else
    for (ctu_y = 0; ctu_y< pic->size->ctu_num_in_height; ctu_y++)
    {     
      for (ctu_x = 0; ctu_x < pic->size->ctu_num_in_width; ctu_x++)
      {
        ctu_address = ctu_y * pic->size->ctu_num_in_width + ctu_x;
#if LAYER_PARALLEL_MEM
		sao_type = sao_params[layer_num][COMP_U][ctu_address].category;   
#else
        sao_type = sao_params[COMP_U][ctu_address].category;   
#endif

        init_ctu_context_for_sao(h265ctx, pic, ctu_address, ctu_y, ctu_x, h265ctx->slice->pps->is_loop_filter_used_across_tile, h265ctx->slice->is_loop_filter_cross_slice);
        if (sao_type > 0)
        {                   
          dst = h265ctx->pic->recon_yuv.buf_u + get_ctu_chroma_offset(h265ctx, 0);
          process_sao_type[sao_type - 1][COMP_U](h265ctx, ctu_y, ctu_x, ctu_address, dst, stride, pic->size, tile, is_tile_boundary_used);
        }   

#if LAYER_PARALLEL_MEM
		sao_type = sao_params[layer_num][COMP_U][ctu_address].category;
#else
        sao_type = sao_params[COMP_U][ctu_address].category;
#endif
        if (sao_type > 0)
        {         
          dst = h265ctx->pic->recon_yuv.buf_v + get_ctu_chroma_offset(h265ctx, 0);
          process_sao_type[sao_type - 1][COMP_V](h265ctx, ctu_y, ctu_x, ctu_address, dst, stride, pic->size, tile, is_tile_boundary_used);
        }   

        if(h265ctx->slice->sps->is_pcm_used && h265ctx->slice->sps->is_pcm_filtering_used || h265ctx->slice->pps->is_bypass_used)
        {
          restore_pcm_chroma(h265ctx, ctu_address, ctu_y, ctu_x);
        }
      }    
    }
#endif
  } 
}
