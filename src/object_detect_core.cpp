# include<ros/ros.h>
# include<sensor_msgs/PointCloud2.h>
# include<pcl/point_cloud.h>// 点云
# include<pcl/point_types.h>// 点云类型
# include<pcl_conversions/pcl_conversions.h>// sensor_msgs::PointCloud2ConstPtr转换成pcl::fromPCLPointCloud2
#include<pcl/filters/voxel_grid.h> //voxel_grid滤波
// # include<pcl/kdtree/kdtree.h> // kdtree
# include<pcl/segmentation/extract_clusters.h>
# include<limits>// 算数极值
# include<jsk_recognition_msgs/BoundingBox.h>// 可视化
# include<jsk_recognition_msgs/BoundingBoxArray.h> 
# include<vector>
# include<stdlib.h>

class lidar_process
{
    private:
        struct detect_rectangle
        {
            pcl::PointXYZ min_points;
            pcl::PointXYZ max_points;
            jsk_recognition_msgs::BoundingBox rectangle;
        };
        ros::Subscriber velodyne_pointcloud;
        ros::Publisher boxes_publish;
        ros::Publisher filter_no_ground;
        std::vector<double> seg_pointcloud_distance,cluster_threshod;// 距离分割,聚类阈值
        std_msgs::Header point_cloud_header;
        void doPointCloud(const sensor_msgs::PointCloud2ConstPtr &in_cloud);
        void point_seg_distance(pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_process_point,std::vector<detect_rectangle> &detection_rectangle_vector);
        void point_seg_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_process_point,double cluster_param,std::vector<detect_rectangle> &detection_rectangle_vector);
    public:
        lidar_process(ros::NodeHandle &nh);
        ~lidar_process();

};

lidar_process::lidar_process(ros::NodeHandle &nh)
{
    seg_pointcloud_distance = {4,8,12,16,30};
    cluster_threshod = {0.2,0.3,0.4,0.5,0.6};
    velodyne_pointcloud = nh.subscribe("/velodyne_points",1000,&lidar_process::doPointCloud,this);
    boxes_publish = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/object_detection_boxes",1000);
    filter_no_ground = nh.advertise<sensor_msgs::PointCloud2>("filter_no_ground",1000);
    ros::spin();
}
lidar_process::~lidar_process() {}

void lidar_process::point_seg_distance(pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_process_point,std::vector<detect_rectangle> &detection_rectangle_vector)
{
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> in_pointcloud(5);
    
    for(auto i = 0;i < in_pointcloud.size();++i)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
        in_pointcloud[i] = tmp;
    }
    for(auto i = 0;i < lidar_process_point->points.size();++i)
    {
        pcl::PointXYZ seg_points;
        seg_points.x = lidar_process_point->points[i].x;
        seg_points.y = lidar_process_point->points[i].y;
        seg_points.z = lidar_process_point->points[i].z;
        float distance = sqrt(pow(seg_points.x,2)+pow(seg_points.y,2));

        if (distance >= seg_pointcloud_distance[4])
        {
            continue;
        }
        if(distance <seg_pointcloud_distance[0])
        {
            in_pointcloud[0]->points.push_back(seg_points);
        }
        else if(distance <seg_pointcloud_distance[1])
        {
            in_pointcloud[1]->points.push_back(seg_points);
        } 
        else if(distance <seg_pointcloud_distance[2])
        {
            in_pointcloud[2]->points.push_back(seg_points);
        }
        else if(distance <seg_pointcloud_distance[3])
        {
            in_pointcloud[3]->points.push_back(seg_points);
        }
        else
        {
            in_pointcloud[4]->points.push_back(seg_points);
        }
            
    }
    for(auto i = 0;i < in_pointcloud.size();++i)
        point_seg_cluster(in_pointcloud[i],cluster_threshod[i],detection_rectangle_vector);
}

void lidar_process::point_seg_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_process_point,double cluster_param,std::vector<detect_rectangle> &detection_rectangle_vector)
{
// 聚类
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_out_points2d(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*lidar_process_point,*plane_out_points2d);
    for(int i = 0;i < plane_out_points2d->points.size();++i)
    {
        plane_out_points2d->points[i].z = 0;
    }
    ROS_INFO("%d",plane_out_points2d->points.size());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);// kd树对象
    if (plane_out_points2d->points.size() > 0)
        kdtree->setInputCloud(plane_out_points2d);//创建点云索引向量，用于存储实际的点云信息
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setInputCloud(plane_out_points2d);// 设置输入点云
    ec.setClusterTolerance(cluster_param);// 搜索半径为cluster_param米
    ec.setMinClusterSize(20);// 设置聚类最小点数
    ec.setMaxClusterSize(100);// 设置聚类最大点数
    ec.setSearchMethod(kdtree); // 设置点云的搜索机制
    ec.extract(cluster_indices);// 从点云中提取聚类，并将点云索引保存在cluster_indices中

    for(auto i = 0;i  < cluster_indices.size();++i)
    {
        detect_rectangle detection_rectangle;
            // 参数初始化
        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double min_z = std::numeric_limits<double>::max();
        double max_x = -std::numeric_limits<double>::max();
        double max_y = -std::numeric_limits<double>::max();
        double max_z = -std::numeric_limits<double>::max();
        for(auto begin = cluster_indices[i].indices.begin();begin != cluster_indices[i].indices.end();++begin)
        {
            pcl::PointXYZ p;
            p.x = lidar_process_point->points[*begin].x;
            p.y = lidar_process_point->points[*begin].y;
            p.z = lidar_process_point->points[*begin].z;

            if(min_x > p.x)
                min_x = p.x;
             if(min_y > p.y)
                min_y = p.y;           
            if(min_z > p.z)
                min_z = p.z;
            if(max_x < p.x)
                max_x = p.x;
            if(max_y <p.y)
                max_y = p.y;         
            if(max_z <p.z)
                max_z = p.z;   
        }
        detection_rectangle.min_points.x = min_x;
        detection_rectangle.min_points.y = min_y;
        detection_rectangle.min_points.z = min_z;
        detection_rectangle.max_points.x = max_x;
        detection_rectangle.max_points.y = max_y;
        detection_rectangle.max_points.z = max_z;

        detection_rectangle.rectangle.header = point_cloud_header;
        
        // 矩形框长宽高
        double length = detection_rectangle.max_points.x - detection_rectangle.min_points.x;
        double width = detection_rectangle.max_points.y - detection_rectangle.min_points.y;
        double height = detection_rectangle.max_points.z - detection_rectangle.min_points.z;

        // 中心点
        detection_rectangle.rectangle.pose.position.x = detection_rectangle.min_points.x + length/2;
        detection_rectangle.rectangle.pose.position.y = detection_rectangle.min_points.y + width/2;
        detection_rectangle.rectangle.pose.position.z = detection_rectangle.min_points.z + height/2;

        // 范围
        detection_rectangle.rectangle.dimensions.x = length;
        detection_rectangle.rectangle.dimensions.y = width;
        detection_rectangle.rectangle.dimensions.z = height;

        detection_rectangle_vector.push_back(detection_rectangle);
    }
}
void lidar_process::doPointCloud(const sensor_msgs::PointCloud2ConstPtr &in_cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr  current_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    point_cloud_header = in_cloud->header;
    pcl::fromROSMsg(*in_cloud,*current_pointcloud);// 点云转换
    // 1.下采样
    pcl::VoxelGrid<pcl::PointXYZ>  voxel_grid;// voxel_grid创建
    voxel_grid.setInputCloud(current_pointcloud);// 设置输入点云
    voxel_grid.setLeafSize(0.2f,0.2f,0.2f);// 设置体素网格大小
    voxel_grid.filter(*voxel_pointcloud);// 设置输出点云

    // 2.RANSAC
    // RANSAC参数初始化
    pcl::PointCloud<pcl::PointXYZ>::Ptr init_plane_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr mid_plane_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_plane_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);// 地面点
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_out_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);// 非地面点
    std::vector<double> plane_param(4);// AX+BY+CZ+D = 0
    std::vector<double> finalPlane_param(4);
    std::vector<int> random_three_points(3);// 随机3个点
    std::vector<double> vector_length(3);// 随机3个点对应向量长度
    double max_plane_distance = -0.2;
    double min_plane_distance =  -1;
    double final_plane_threshold = 0.2;
    double p =0.8;//RANSAC参数
    int max_points_num = 0;
    int iter_Maxcount = 100;// RANSAC计算更新
    int init_count = 0;
    

    // 初步筛选地面点
    for(int i = 0;i < voxel_pointcloud->points.size(); ++i)
    {
        if(voxel_pointcloud->points[i].z <= max_plane_distance && voxel_pointcloud->points[i].z >= min_plane_distance)
        {
            init_plane_pointcloud->points.push_back(voxel_pointcloud->points[i]);
        }
    }

    // RANSAC
    while(init_count < iter_Maxcount)
    {
        mid_plane_pointcloud->points.clear();
        for(int i = 0;i < 3;++i)
        {
            random_three_points[i] = rand() % (init_plane_pointcloud->points.size());
        }
        // 判断3个点是否相同
        double x1 = init_plane_pointcloud->points[random_three_points[0]].x;
        double y1 = init_plane_pointcloud->points[random_three_points[0]].y;
        double z1 = init_plane_pointcloud->points[random_three_points[0]].z;
        double x2 = init_plane_pointcloud->points[random_three_points[1]].x;
        double y2 = init_plane_pointcloud->points[random_three_points[1]].y;
        double z2 = init_plane_pointcloud->points[random_three_points[1]].z;
        double x3 = init_plane_pointcloud->points[random_three_points[2]].x;
        double y3 = init_plane_pointcloud->points[random_three_points[2]].y;
        double z3 = init_plane_pointcloud->points[random_three_points[2]].z;

        vector_length[0] = (x2-x1)*(x3-x1)+(y2-y1)*(y3-y1)+(z2-z1)*(z3-z1);
        vector_length[1] = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1) + (z2 - z1)*(z2 - z1));// (x1,y1,z1)与(x2,y2,z2)距离
        vector_length[2] = sqrt((x3 - x1)*(x3 - x1) + (y3 - y1)*(y3 - y1) + (z3 - z1)*(z3 - z1));// (x1,y1,z1)与(x3,y3,z3)距离
        double vector_length_angleValue =  vector_length[0] /(vector_length[1] * vector_length[2] );

        if(vector_length_angleValue == 1)
        {
            continue;
        }
        //A B C D
        plane_param[0] = (y2-y1)*(z3-z1)-(y3-y1)*(z2-z1);// A
        plane_param[1] = (z2 - z1)*(x3 - x1) - (x2 - x1)*(z3 - z1);// B
        plane_param[2] = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1); // C
        plane_param[3] = -(plane_param[0]*x1+plane_param[1]*y1+plane_param[2]*z1);// D

        // 点集合
        for(int i = 0;i < init_plane_pointcloud->points.size();++i)
        {
            double A = plane_param[0];
            double B = plane_param[1];
            double C = plane_param[2];
            double D = plane_param[3];
            double X = init_plane_pointcloud->points[i].x;
            double Y = init_plane_pointcloud->points[i].y;
            double Z = init_plane_pointcloud->points[i].z;
            double point_to_plane = abs(A*X+B*Y+C*Z+D)/sqrt(A*A+B*B+C*C);
            if (point_to_plane < final_plane_threshold)
            { 
                mid_plane_pointcloud->points.push_back(init_plane_pointcloud->points[i]);
            }
        }
        if(mid_plane_pointcloud->points.size() > max_points_num)
        {
            final_plane_pointcloud->points.clear();
            pcl::copyPointCloud(*mid_plane_pointcloud,*final_plane_pointcloud);
            max_points_num = mid_plane_pointcloud->points.size();
            for(int i = 0;i < 4;++i)
            {
                finalPlane_param[i] = plane_param[i];
            }
            double w = max_points_num/init_plane_pointcloud->points.size();
            iter_Maxcount = log(1-p)/log(1-pow(w,3));
        }
            init_count ++;
        if(init_count > iter_Maxcount)
        {
            break;
        }
    }
    // 非地面点
    for(int i = 0;i < voxel_pointcloud->points.size();++i)
    {
        double A = finalPlane_param[0];
        double B = finalPlane_param[1];
        double C = finalPlane_param[2];
        double D = finalPlane_param[3];
        double X = voxel_pointcloud->points[i].x;
        double Y = voxel_pointcloud->points[i].y;
        double Z = voxel_pointcloud->points[i].z;
        double point_to_plane = abs(A*X+B*Y+C*Z+D)/sqrt(A*A+B*B+C*C);
        if(point_to_plane >= final_plane_threshold)
        {
            plane_out_pointcloud->points.push_back(voxel_pointcloud->points[i]);
        }
    }
    
    // 框
    std::vector<detect_rectangle> detection_rectangle_vector;
    jsk_recognition_msgs::BoundingBoxArray box;
    point_seg_distance(plane_out_pointcloud,detection_rectangle_vector);
    for(int i = 0;i < detection_rectangle_vector.size();++i)
    {
        box.boxes.push_back(detection_rectangle_vector[i].rectangle);
    }
    box.header = in_cloud->header; 
    boxes_publish.publish(box);

    // 非地面点可视化
    sensor_msgs::PointCloud2 filter_points;
    pcl::toROSMsg(*plane_out_pointcloud,filter_points);
    filter_points.header = point_cloud_header;
    filter_no_ground.publish(filter_points);

}

int main(int argc,char *argv[])
{
    setlocale(LC_ALL,"");
    ROS_INFO("数据处理中,输出检测框");
    ros::init(argc,argv,"object_detection");
    ros::NodeHandle nh;
    lidar_process core(nh);
    return 0;
}