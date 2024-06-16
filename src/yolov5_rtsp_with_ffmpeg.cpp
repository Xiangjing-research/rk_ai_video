/**
 * OpenCv捕获数据，通过FFmpeg的编码器将数据发出
 */
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <rockchip/mpp_buffer.h>
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_meta.h>
#include <libavutil/hwcontext_drm.h>
#include <ffmpeg_with_mpp.h>
#include <opencv2/videoio.hpp>

/*-------------------------------------------
                Includes
-------------------------------------------*/

#include <rknn_func.h>
#include <config.h>

#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
using namespace std;
using namespace cv;

unsigned int framecount = 0;
unsigned int width = 0, height = 0;
/**
 * 分别对应水平补齐后的步长和垂直补齐后的步长
 * RK3588 读取图片时是16字节对齐，因此需要对图片补齐方能处理
 */
unsigned int hor_stride = 0, ver_stride = 0;

unsigned int yuv_width = 0, yuv_height = 0;
unsigned int yuv_hor_stride = 0, yuv_ver_stride = 0;

/**
 * 1920x1080x3 的RGB图片补齐后
 * hor_stride 1920,ver_stride = 1088
 * 对于YUV图像，图像会下采样成1920x1080x3/2x1(Channel)
 * yuv_hor_stride 1920,yuv_ver_stride = 1632
 * 这两个图片的大小从
 * 1920x1080x3(rgb) --> 1920x1080x3/2(yuv)
 * wxhxc:
 * 1920x1080x3(rgb) --> 1920x1620x1(yuv)
 * 同时由于补齐作用 YUV原始图像需要变成:
 * 1920x1620 --> 1920x1632
 * YUV图像分量的分布为：wxh的亮度Y，w/2 x h/2的U，w/2 x h/2的V
 * -------------w(1920)----------
 * |                            |
 * |                            |
 * |                            |
 * |             Y              h(1080)
 * |                            |
 * |                            |
 * |                            |
 * ------------------------------
 * |                            |
 * |             gap            | 8
 * ------------------------------
 * |                            |
 * |              U             h/2(540)
 * |                            |
 * ------------------------------
 * |             gap            | 4
 * ------------------------------
 * |                            |
 * |              V             h/2(540)
 * |                            |
 * ------------------------------
 * |             gap            | 4
 * ------------------------------
 *
 */

unsigned int image_size = 0;

/*********************FFMPEG_START*/
const AVCodec *codec;
AVCodecContext *codecCtx;
AVFormatContext *formatCtx;
AVStream *stream;
AVHWDeviceType type = AV_HWDEVICE_TYPE_DRM;
AVBufferRef *hwdevice;
AVBufferRef *hwframe;
AVHWFramesContext *hwframeCtx;

AVFrame *frame;   // 封装DRM的帧
AVPacket *packet; // 发送的包

long extra_data_size = 10000000;
uint8_t *cExtradata = NULL;

AVPixelFormat hd_pix = AV_PIX_FMT_DRM_PRIME;
AVPixelFormat sw_pix = AV_PIX_FMT_YUV420P;
/*********************FFMPEG_END*/
/**********************MPP_START*/
MppBufferGroup group;
MppBufferInfo info;
MppBuffer buffer;
MppBuffer commitBuffer;
MppFrame mppframe;
MppPacket mppPacket;
typedef struct
{
    MppFrame frame;
    AVBufferRef *decoder_ref;
} RKMPPFrameContext;
/************************MPP_END*/
std::map<std::string, std::string> configMap;

void rkmpp_release_frame(void *opaque, uint8_t *data)
{
    AVDRMFrameDescriptor *desc = (AVDRMFrameDescriptor *)data;
    AVBufferRef *framecontextref = (AVBufferRef *)opaque;
    RKMPPFrameContext *framecontext = (RKMPPFrameContext *)framecontextref->data;

    mpp_frame_deinit(&framecontext->frame);
    av_buffer_unref(&framecontext->decoder_ref);
    av_buffer_unref(&framecontextref);

    av_free(desc);
}

int init_encoder(config &cfg)
{
    int res = 0;
    avformat_network_init();

    codec = avcodec_find_encoder_by_name("h264_rkmpp");
    if (!codec)
    {
        print_error(__LINE__, -1, "can not find h264_rkmpp encoder!");
        return -1;
    }
    // 创建编码器上下文
    codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx)
    {
        print_error(__LINE__, -1, "can not create codec Context of h264_rkmpp!");
        return -1;
    }

    res = av_hwdevice_ctx_create(&hwdevice, type, "/dev/dri/card0", 0, 0);
    if (res < 0)
    {
        print_error(__LINE__, res, "create hdwave device context failed!");
        return res;
    }

    codecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    codecCtx->codec_id = codec->id;
    codecCtx->codec = codec;
    codecCtx->bit_rate = 1024 * 1024 * 8;
    codecCtx->codec_type = AVMEDIA_TYPE_VIDEO; // 解码类型
    codecCtx->width = width;                   // 宽
    codecCtx->height = height;                 // 高
    codecCtx->channel_layout = 0;
    codecCtx->time_base = (AVRational){1, cfg.get_fps()}; // 每帧的时间
    codecCtx->framerate = (AVRational){cfg.get_fps(), 1}; // 帧率v b
    codecCtx->pix_fmt = hd_pix;                           // AV_PIX_FMT_DRM_PRIME
    codecCtx->gop_size = 12;                              // 每组多少帧
    codecCtx->max_b_frames = 0;                           // b帧最大间隔

    hwframe = av_hwframe_ctx_alloc(hwdevice);
    if (!hwframe)
    {
        print_error(__LINE__, -1, "create hdwave frame context failed!");
        return -1;
    }
    hwframeCtx = (AVHWFramesContext *)(hwframe->data);
    hwframeCtx->format = hd_pix;
    hwframeCtx->sw_format = sw_pix;
    hwframeCtx->width = width;
    hwframeCtx->height = height;
    /**
     *  帧池，会预分配，后面创建与硬件关联的帧时，会从该池后面获取相应的帧
     *  initial_pool_size与pool 至少要有一个不为空
     */
    // hwframeCtx->initial_pool_size = 20;
    hwframeCtx->pool = av_buffer_pool_init(20 * sizeof(AVFrame), NULL);
    res = av_hwframe_ctx_init(hwframe);
    if (res < 0)
    {
        print_error(__LINE__, res, "init hd frame context failed!");
        return res;
    }
    codecCtx->hw_frames_ctx = hwframe;
    codecCtx->hw_device_ctx = hwdevice;

    if (!strcmp(cfg.get_protocol(), "rtsp"))
    {
        // rtsp协议
        res = avformat_alloc_output_context2(&formatCtx, NULL, "rtsp", cfg.get_url());
    }
    else
    {
        // rtmp协议
        res = avformat_alloc_output_context2(&formatCtx, NULL, "flv", cfg.get_url());
    }
    if (res < 0)
    {
        print_error(__LINE__, res, "create output context failed!");
        return res;
    }

    stream = avformat_new_stream(formatCtx, codec);
    if (!stream)
    {
        print_error(__LINE__, res, "create stream failed!");
        return -1;
    }
    stream->time_base = (AVRational){1, cfg.get_fps()}; // 设置帧率
    stream->id = formatCtx->nb_streams - 1;             // 设置流的索引
    stream->codecpar->codec_tag = 0;

    res = avcodec_parameters_from_context(stream->codecpar, codecCtx);
    if (res < 0)
    {
        print_error(__LINE__, res, "copy parameters to stream failed!");
        return -1;
    }

    // 打开输出IO RTSP不需要打开，RTMP需要打开
    if (!strcmp(cfg.get_protocol(), "rtmp"))
    {
        res = avio_open2(&formatCtx->pb, cfg.get_url(), AVIO_FLAG_WRITE, NULL, NULL);
        if (res < 0)
        {
            print_error(__LINE__, res, "avio open failed !");
            return -1;
        }
    }
    // 写入头信息
    AVDictionary *opt = NULL;
    if (!strcmp(cfg.get_protocol(), "rtsp"))
    {
        av_dict_set(&opt, "rtsp_transport", cfg.get_trans_protocol(), 0);
        av_dict_set(&opt, "muxdelay", "0.1", 0);
    }
    res = avformat_write_header(formatCtx, &opt);
    if (res < 0)
    {
        print_error(__LINE__, res, "avformat write header failed ! ");
        return -1;
    }
    av_dump_format(formatCtx, 0, cfg.get_url(), 1);

    // open codec
    AVDictionary *opencodec = NULL;
    av_dict_set(&opencodec, "preset", cfg.get_preset(), 0);
    av_dict_set(&opencodec, "tune", cfg.get_tune(), 0);
    av_dict_set(&opencodec, "profile", cfg.get_profile(), 0);
    // 打开编码器
    res = avcodec_open2(codecCtx, codec, &opencodec);
    if (res < 0)
    {
        print_error(__LINE__, res, "open codec failed ! ");
        return -1;
    }
    return res;
}

MPP_RET init_mpp()
{
    MPP_RET res = MPP_OK;
    res = mpp_buffer_group_get_external(&group, MPP_BUFFER_TYPE_DRM);
    return res;
}

int init_data(config &cfg)
{
    int res = 0;
    // 给packet 分配内存
    packet = av_packet_alloc();

    /**
     * 初始化包空间，进行PPS SPS包头填充
     * 使用h264_rkmpp编码器时，rtsp/rtmp协议都需要添加PPS
     * libx264只需要在rtsp协议时添加PPS,rtmp会自动加上
     */
    cExtradata = (uint8_t *)malloc((extra_data_size) * sizeof(uint8_t));

    frame = av_frame_alloc(); // 分配空间
    frame->width = width;
    frame->height = height;
    frame->format = sw_pix;
    res = av_hwframe_get_buffer(codecCtx->hw_frames_ctx, frame, 0); // 与硬件帧创建关联

    if (!frame->hw_frames_ctx)
    {
        print_error(__LINE__, res, " connect frame to hw frames ctx failed!");
    }
    return res;
}

MPP_RET read_frame(cv::Mat &cvframe, void *ptr)
{
    RK_U32 row = 0;
    RK_U32 read_size = 0;
    RK_U8 *buf_y = (RK_U8 *)ptr;
    RK_U8 *buf_u = buf_y + hor_stride * ver_stride;     // NOTE: diff from gen_yuv_image
    RK_U8 *buf_v = buf_u + hor_stride * ver_stride / 4; // NOTE: diff from gen_yuv_image
    // buf_y = cvframe.data;

    for (row = 0; row < height; row++)
    {
        memcpy(buf_y + row * hor_stride, cvframe.datastart + read_size, width);
        read_size += width;
    }

    for (row = 0; row < height / 2; row++)
    {
        memcpy(buf_u + row * hor_stride / 2, cvframe.datastart + read_size, width / 2);
        read_size += width / 2;
    }

    for (row = 0; row < height / 2; row++)
    {
        memcpy(buf_v + row * hor_stride / 2, cvframe.datastart + read_size, width / 2);
        read_size += width / 2;
    }
    return MPP_OK;
}

int send_packet(config &cfg)
{
    int res = 0;
    packet->pts = av_rescale_q_rnd(framecount, codecCtx->time_base, stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_NEAR_INF));
    packet->dts = av_rescale_q_rnd(framecount, codecCtx->time_base, stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_NEAR_INF));
    packet->duration = av_rescale_q_rnd(packet->duration, codecCtx->time_base, stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_NEAR_INF));

    if (!(packet->flags & AV_PKT_FLAG_KEY))
    {
        // 在每帧非关键帧前面添加PPS SPS头信息
        /**
         * 使用h264_rkmpp编码器时，rtsp/rtmp协议都需要添加PPS
         * libx264只需要在rtsp协议时添加PPS,rtmp会自动加上
         */
        int packet_data_size = packet->size;
        u_char frame_data[packet_data_size];
        memcpy(frame_data, packet->data, packet->size);
        memcpy(packet->data, cExtradata, extra_data_size);
        memcpy(packet->data + extra_data_size, frame_data, packet_data_size);
        packet->size = packet_data_size + extra_data_size;
    }
    // 通过创建输出流的format 输出数据包
    framecount++;
    res = av_interleaved_write_frame(formatCtx, packet);
    if (res < 0)
    {
        print_error(__LINE__, res, "send packet error!");
        return -1;
    }
    return 0;
}
/**
 * 将opencv的帧转换成Drm数据帧
 */
MPP_RET convert_cvframe_to_drm(cv::Mat &cvframe, AVFrame *&avframe)
{
    MPP_RET res = MPP_OK;
    res = mpp_buffer_get(NULL, &buffer, image_size);
    if (res != MPP_OK)
    {
        print_error(__LINE__, res);
        return res;
    }
    info.fd = mpp_buffer_get_fd(buffer);
    info.ptr = mpp_buffer_get_ptr(buffer);
    info.index = framecount;
    info.size = image_size;
    info.type = MPP_BUFFER_TYPE_DRM;
    // 将数据读入buffer
    read_frame(cvframe, info.ptr);

    res = mpp_buffer_commit(group, &info);
    if (res != MPP_OK)
    {
        return res;
    }

    res = mpp_buffer_get(group, &commitBuffer, image_size);
    if (res != MPP_OK)
    {
        return res;
    }

    mpp_frame_init(&mppframe);
    mpp_frame_set_width(mppframe, width);
    mpp_frame_set_height(mppframe, height);
    mpp_frame_set_hor_stride(mppframe, yuv_hor_stride);
    mpp_frame_set_ver_stride(mppframe, ver_stride);
    mpp_frame_set_buf_size(mppframe, image_size);
    mpp_frame_set_buffer(mppframe, commitBuffer);
    /**
     * 使用mpp可以使用 YUV格式的数据外 还能使用RGB格式的数据
     * 但是使用ffmpeg的h264_rkmpp系列编码器只能使用YUV格式
     */
    mpp_frame_set_fmt(mppframe, MPP_FMT_YUV420SP); // YUV420SP == NV12
    mpp_frame_set_eos(mppframe, 0);

    AVDRMFrameDescriptor *desc = (AVDRMFrameDescriptor *)av_mallocz(sizeof(AVDRMFrameDescriptor));
    if (!desc)
    {
        return MPP_NOK;
    }
    desc->nb_objects = 1;
    desc->objects[0].fd = mpp_buffer_get_fd(commitBuffer);
    desc->objects[0].size = mpp_buffer_get_size(commitBuffer);

    desc->nb_layers = 1;
    AVDRMLayerDescriptor *layer = &desc->layers[0];
    layer->format = DRM_FORMAT_YUV420;
    layer->nb_planes = 2;

    // Y 分量
    layer->planes[0].object_index = 0;
    layer->planes[0].offset = 0;
    layer->planes[0].pitch = mpp_frame_get_hor_stride(mppframe); // 1920

    // 第二层分量
    layer->planes[1].object_index = 0;
    layer->planes[1].offset = layer->planes[0].pitch * ver_stride; // 1920 * 1088
    layer->planes[1].pitch = layer->planes[0].pitch;

    avframe->reordered_opaque = avframe->pts;

    avframe->color_range = (AVColorRange)mpp_frame_get_color_range(mppframe);
    avframe->color_primaries = (AVColorPrimaries)mpp_frame_get_color_primaries(mppframe);
    avframe->color_trc = (AVColorTransferCharacteristic)mpp_frame_get_color_trc(mppframe);
    avframe->colorspace = (AVColorSpace)mpp_frame_get_colorspace(mppframe);

    auto mode = mpp_frame_get_mode(mppframe);
    avframe->interlaced_frame = ((mode & MPP_FRAME_FLAG_FIELD_ORDER_MASK) == MPP_FRAME_FLAG_DEINTERLACED);
    avframe->top_field_first = ((mode & MPP_FRAME_FLAG_FIELD_ORDER_MASK) == MPP_FRAME_FLAG_TOP_FIRST);

    AVBufferRef *framecontextref = (AVBufferRef *)av_buffer_allocz(sizeof(AVBufferRef));
    if (!framecontextref)
    {
        return MPP_NOK;
    }

    // MPP decoder needs to be closed only when all frames have been released.
    RKMPPFrameContext *framecontext = (RKMPPFrameContext *)framecontextref->data;
    framecontext->frame = mppframe;

    avframe->data[0] = (uint8_t *)desc;
    avframe->buf[0] = av_buffer_create((uint8_t *)desc, sizeof(*desc), rkmpp_release_frame,
                                       framecontextref, AV_BUFFER_FLAG_READONLY);

    return res;
}

int transfer_frame(cv::Mat &cvframe, config &cfg,int num_frame)
{
    
    int packetFinish = 0, frameFinish = 0;
    int res = 0;
    // 给帧打上时间戳
    frame->pts = (framecount)*av_q2d(codecCtx->time_base);
    // 一行（宽）数据的字节数 列数x3

    res = convert_cvframe_to_drm(cvframe, frame);

    if (res < 0)
    {
        print_error(__LINE__, res, "transfer data to hdwave failed !");
        return -1;
    }
    res = avcodec_send_frame(codecCtx, frame);
    if (res != 0)
    {
        print_error(__LINE__, res, "send frame to avcodec failed !");
        return -1;
    }

    res = avcodec_receive_packet(codecCtx, packet);
    if (res != 0)
    {
        print_error(__LINE__, res, "fail receive encode packet!");
        return -1;
    }
    send_packet(cfg);

    if (buffer != NULL)
    {
        mpp_buffer_put(buffer); // 清空buffer
        buffer = NULL;
    }
    if (commitBuffer != NULL)
    {
        mpp_buffer_put(commitBuffer); // 清空buffer
        commitBuffer = NULL;
    }
    mpp_buffer_group_clear(group);
    mpp_frame_deinit(&mppframe);
    printf("transfer frame:%d\n",  num_frame);
    return 0;
}

void destory_()
{
    cout << "释放回收资源：" << endl;
    mpp_buffer_group_put(group);
    // fclose(wf);
    if (formatCtx)
    {
        // avformat_close_input(&avFormatCtx);
        avio_close(formatCtx->pb);
        avformat_free_context(formatCtx);
        formatCtx = 0;
        cout << "avformat_free_context(formatCtx)" << endl;
    }
    if (packet)
    {
        av_packet_unref(packet);
        packet = NULL;
        cout << "av_free_packet(packet)" << endl;
    }

    if (frame)
    {
        av_frame_free(&frame);
        frame = 0;
        cout << "av_frame_free(frame)" << endl;
    }

    if (codecCtx->hw_device_ctx)
    {
        av_buffer_unref(&codecCtx->hw_device_ctx);
        cout << "av_buffer_unref(&codecCtx->hw_device_ctx)" << endl;
    }

    if (codecCtx)
    {
        avcodec_close(codecCtx);
        codecCtx = 0;
        cout << "avcodec_close(codecCtx);" << endl;
    }
    if (cExtradata)
    {
        free(cExtradata);
        cout << "free cExtradata " << endl;
    }
    if (hwdevice)
    {
        av_buffer_unref(&hwdevice);
        cout << "av_buffer_unref hwdevice " << endl;
    }
    if (hwframe)
    {
        av_buffer_unref(&hwframe);
        cout << "av_buffer_unref encodeHwBufRef " << endl;
    }
}
/**
 * 初始化h264_rkmpp编码器
 */
int init_h264_rkmpp_encoder(config &cfg, cv::Mat &cvframe, cv::Mat &yuvframe)
{
    Size size_yuv = yuvframe.size();
    yuv_width = size_yuv.width;
    yuv_height = size_yuv.height;

    yuv_hor_stride = MPP_ALIGN(yuv_width, 16);  // 1920
    yuv_ver_stride = MPP_ALIGN(yuv_height, 16); // 1632

    Size size = cvframe.size();
    width = size.width;
    height = size.height;

    hor_stride = MPP_ALIGN(width, 16);  // 1920
    ver_stride = MPP_ALIGN(height, 16); // 1088

    image_size = sizeof(unsigned char) * yuv_hor_stride * yuv_hor_stride;
    cout << " width " << width << endl;
    cout << " height " << height << endl;
    cout << " hor_stride " << hor_stride << endl;
    cout << " ver_stride " << ver_stride << endl;
    // cout << format(yuvframe,Formatter::FMT_C) << endl;
    cout << " yuv frame info " << endl;
    cout << " yuv frame cols " << yuvframe.cols << endl;
    cout << " yuv frame rows " << yuvframe.rows << endl;
    cout << " yuv frame elesize " << yuvframe.elemSize() << endl;
    cout << " yuv frame channel " << yuvframe.channels() << endl;
    int res = init_encoder(cfg); // 初始化解码器
    if (res < 0)
    {
        print_error(__LINE__, res, "init encoder failed!");
    }
    res = init_data(cfg);
    if (res < 0)
    {
        print_error(__LINE__, res, "init data failed!");
    }
    res = init_mpp();
    if (res < 0)
    {
        print_error(__LINE__, res, "init mpp failed!");
    }
    return res;
}

void ConfigFileRead(map<string, string> &m_mapConfigInfo, string path = "./config.conf")
{
    ifstream configFile;
    configFile.open(path.c_str());
    string str_line;
    if (configFile.is_open())
    {
        while (!configFile.eof())
        {
            getline(configFile, str_line);
            if (str_line.compare(0, 1, "#") == 0)
            {
                continue;
            }
            size_t pos = str_line.find('=');
            string str_key = str_line.substr(0, pos);
            string str_value = str_line.substr(pos + 1);
            m_mapConfigInfo.insert(pair<string, string>(str_key, str_value));
        }
    }
    else
    {
        cout << "Cannot open config file config.conf, path: ";
        exit(-1);
    }
}

/**
 * 交由FFmpeg完成硬件编码,
 * FFmpeg由于是魔改版本，封装的硬件解码器只支持输入Drm帧的格式，因此需要将帧数据封装成AVFrame。并送入AvCodec，然后得到AvPacket,最后发送。整体流程如下：
----------                     ---------                     ----------                    --------                     --------
| opencv | -->frame(BGR2RGB)-->| model | -->frame(RGB2YUV)-->| func | -->AvFrame(drm)--> | ffmpeg | -->encode(mpp)--> | send | -->rtsp
----------                     ---------                     ----------                    --------                     --------
 *
*/

int main(int argc, char *argv[])
{

    // Command obj = process_command(argc, argv);
    map<string, string> mapConfigInfo;
    ConfigFileRead(mapConfigInfo, "./config.conf");
    config cfg(mapConfigInfo);
    // int res = 0;
    VideoCapture videoCap;
    // videoCap.set(CAP_PROP_FRAME_WIDTH, 1920);//宽度
    // videoCap.set(CAP_PROP_FRAME_HEIGHT, 1080);//高度
    // videoCap.set(CAP_PROP_FPS, 30);//帧率 帧/秒
    // videoCap.set(CAP_PROP_FOURCC,VideoWriter::fourcc('M','J','P','G')); // 捕获格式
    // videoCap.set(CAP_PROP_FOURCC,VideoWriter::fourcc('I','4','2','0')); // 捕获格式
    // videoCap.set(CAP_PROP_CONVERT_RGB,0);

    /*------------------------------初始化rknn-----------------------------------------*/
    int ret;
    rknn_context ctx;
    size_t actual_size = 0;
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
    struct timeval start_time, stop_time;
    int option = 1;

    // init rga context
    rga_buffer_t src;
    rga_buffer_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

    /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model(cfg.get_model_name(), &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);
    // ======================通用接口输出形式======================
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // ======================通用接口输出形式======================

    /*---------------------------零拷贝实现-输入/输出内存由运行时分配---------------------------------------*/
    //======================零拷贝接口输入形式======================
    // rknn_tensor_mem *input_mems[1];
    // 这里有个有意思的现象，这里模型输入的type格式默认为RKNN_TENSOR_INT8，这就意味着，
    // 归一化及量化操作要在CPU侧进行处理，也就是读完数据后就进行操作，而如果
    // 设置为RKNN_TENSOR_UINT8则归一化及量化操作都放到了NPU上进行。
    // input_attrs[0].type = RKNN_TENSOR_UINT8;
    // input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    // ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    // if (ret < 0)
    // {
    //     printf("rknn_set_io_mem fail! ret=%d\n", ret);
    //     return -1;
    // }
    //======================零拷贝接口输出形式======================
    // rknn_tensor_mem *output_mems[1];
    // for (uint32_t i = 0; i < io_num.n_output; ++i)
    // {
    // int output_size = output_attrs[i].n_elems * sizeof(float);
    // output_mems[i] = rknn_create_mem(ctx, output_size);
    // }
    // for (uint32_t i = 0; i < io_num.n_output; ++i)
    // {
    //     output_attrs[i].type = RKNN_TENSOR_FLOAT32; // 这里设置float32，反量化操作在NPU内进行。
    //     int ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
    //     if (ret < 0)
    //     {
    //         printf("rknn_set_io_mem fail! ret=%d\n", ret);
    //         return -1;
    //     }
    // }

    /*-----------------------------------------------------------------------------*/
    /*------------------------------读取OpenCV-----------------------------------------*/
    // videoCap.open(R"(/home/orangepi/ubuntu_desktop/rk_ai_video/720p60hz.h264)");
    videoCap.open(cfg.get_input_path());
    if (!videoCap.isOpened())
    {
        cout << "video_file not open !" << endl;
        return -1;
    }

    int is_init_encoder = 0;
    Mat cv_frame, rgb_frame, yuv_frame;
    int num_frame = 0;
    while (videoCap.read(cv_frame))
    {
        if (!cv_frame.data)
        {
            continue;
        }
        // 转换图片格式
        cvtColor(cv_frame, rgb_frame, cv::COLOR_BGR2RGB);
        // // img_width = rgb_frame.cols;
        // // img_height = rgb_frame.rows;
        // // printf("img width = %d, img height = %d\n", img_width, img_height);

        // /*----------------------------preprocess_rga------------------------------------*/
        // // 指定目标大小和预处理方式,默认使用LetterBox的预处理
        // BOX_RECT pads;
        // memset(&pads, 0, sizeof(BOX_RECT));
        // cv::Size target_size(width, height);
        // cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
        // // 计算缩放比例
        // float scale_w = (float)target_size.width / rgb_frame.cols;
        // float scale_h = (float)target_size.height / rgb_frame.rows;

        // if (img_width != width || img_height != height)
        // {
        //     // 直接缩放采用RGA加速
        //     // 直接对图像进行resize，改变了图像的长宽比，图像会拉伸
        //     if (option == 0)
        //     {
        //         printf("resize image by rga\n");
        //         ret = resize_rga(src, dst, rgb_frame, resized_img, target_size);
        //         if (ret != 0)
        //         {
        //             fprintf(stderr, "resize with rga error\n");
        //             return -1;
        //         }
        //         // 保存预处理图片
        //         // cv::imwrite("resize_input.jpg", resized_img);
        //     }
        //     // 图像在保持长宽比的情况下，填充到一个盒子内，在短边上去填充0像素，使得图像扩充为网络输入尺寸
        //     else if (option == 1)
        //     {
        //         printf("resize image with letterbox\n");
        //         float min_scale = std::min(scale_w, scale_h);
        //         scale_w = min_scale;
        //         scale_h = min_scale;
        //         letterbox(rgb_frame, resized_img, pads, min_scale, target_size);
        //         // 保存预处理图片
        //         // cv::imwrite("letterbox_input.jpg", resized_img);
        //         // cv::imwrite("/home/orangepi/ubuntu_desktop/rk_ai_video/install/rk_ffmpeg/output/letterbox_input_" + to_string(num_frame) + ".jpg", resized_img);
        //     }
        //     else
        //     {
        //         fprintf(stderr, "Invalid resize option. Use 'resize' or 'letterbox'.\n");
        //         return -1;
        //     }
        //     // 数据更新
        //     inputs[0].buf = resized_img.data;
        //     // 数据更新-零拷贝实现
        //     // memcpy(input_mems[0]->virt_addr, resized_img.data, input_attrs[0].size_with_stride);
        // }
        // else
        // {
        //     // 数据更新
        //     inputs[0].buf = rgb_frame.data;
        //     // 数据更新-零拷贝实现
        //     // memcpy(input_mems[0]->virt_addr, rgb_frame.data, input_attrs[0].size_with_stride);
        // }

        // gettimeofday(&start_time, NULL);
        // // ======================通用接口输出形式======================
        // rknn_inputs_set(ctx, io_num.n_input, inputs);

        // rknn_output outputs[io_num.n_output];
        // memset(outputs, 0, sizeof(outputs));
        // for (int i = 0; i < io_num.n_output; i++)
        // {
        //     outputs[i].want_float = 0;
        // }
        // // ======================通用接口输出形式======================
        // 初始化h264_rkmpp_encoder
        if (!is_init_encoder)
        {
            if (init_h264_rkmpp_encoder(cfg, cv_frame, rgb_frame) < 0)
            {
                break;
            }
            else
            {
                is_init_encoder = 1;
            }
        }
        // /*------------------------------推理-----------------------------------------*/
        // // 执行推理
        // ret = rknn_run(ctx, NULL);
        // // 获取输出-通用api
        // ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        // gettimeofday(&stop_time, NULL);
        // printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

        // /*------------------------------后处理-----------------------------------------*/
        // detect_result_group_t detect_result_group;
        // std::vector<float> out_scales;
        // std::vector<int32_t> out_zps;
        // for (int i = 0; i < io_num.n_output; ++i)
        // {
        //     out_scales.push_back(output_attrs[i].scale);
        //     out_zps.push_back(output_attrs[i].zp);
        // }
        // // post_process((int8_t *)output_mems[0]->virt_addr, (int8_t *)output_mems[1]->virt_addr, (int8_t *)output_mems[2]->virt_addr, height, width,
        // //              box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
        // post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
        //              box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
        // // 画框和概率
        // char text[256];
        // for (int i = 0; i < detect_result_group.count; i++)
        // {
        //     detect_result_t *det_result = &(detect_result_group.results[i]);
        //     sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        //     printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
        //            det_result->box.right, det_result->box.bottom, det_result->prop);
        //     int x1 = det_result->box.left;
        //     int y1 = det_result->box.top;
        //     int x2 = det_result->box.right;
        //     int y2 = det_result->box.bottom;
        //     rectangle(rgb_frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(256, 0, 0, 256), 3);
        //     putText(rgb_frame, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        // }
        // // cv::imwrite("./output/infer_" + to_string(num_frame) + ".jpg", resized_img);
        // /*------------------------------rtsp推流-----------------------------------------*/
        // // rgb_frame2yuv
        cvtColor(rgb_frame, yuv_frame, cv::COLOR_BGR2RGB);
        // 封装成drm帧
        transfer_frame(yuv_frame, cfg,num_frame);
        av_packet_unref(packet);
        num_frame++;
    }

    /*------------------------------release-----------------------------------------*/
    deinitPostProcess();

    ret = rknn_destroy(ctx);

    // for (int i = 0; i < io_num.n_input; i++)
    // {
    //     rknn_destroy_mem(ctx, input_mems[i]);
    // }

    // for (int i = 0; i < io_num.n_output; i++)
    // {
    //     rknn_destroy_mem(ctx, output_mems[i]);
    // }

    // RGA_deinit(&rga_ctx);
    if (model_data)
    {
        free(model_data);
    }

    if (cExtradata)
    {
        free(cExtradata);
    }

FAIL:
    videoCap.release();
    destory_();
    return 0;
}