#include "utility.h"
#include "lio_sam/cloud_info.h"

struct VelodynePointXYZIRT
{
	PCL_ADD_POINT4D									// 位置
			PCL_ADD_INTENSITY;					// 激光点反射强度 float intensity;
	uint16_t ring;									// 扫描线
	float time;											// 时间戳，记录相对于当前帧第一个激光点的时差，第一个点time=0
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 内存16字节对齐，EIGEN SSE优化要求
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(VelodynePointXYZIRT,
																	(float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint16_t, ring, ring)(float, time, time))

struct OusterPointXYZIRT
{
	PCL_ADD_POINT4D;
	float intensity;
	uint32_t t;
	uint16_t reflectivity;
	uint8_t ring;
	uint16_t noise;
	uint32_t range;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
																	(float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, t, t)(uint16_t, reflectivity, reflectivity)(uint8_t, ring, ring)(uint16_t, noise, noise)(uint32_t, range, range))

struct MulranPointXYZIRT
{ // from the file player's topic https://github.com/irapkaist/file_player_mulran, see https://github.com/irapkaist/file_player_mulran/blob/17da0cb6ef66b4971ec943ab8d234aa25da33e7e/src/ROSThread.cpp#L7
	PCL_ADD_POINT4D;
	float intensity;
	uint32_t t;
	int ring;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(MulranPointXYZIRT,
																	(float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, t, t)(int, ring, ring))

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;
// IMU数据队列长度
const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:
	// IMU队列、odom队列互斥锁
	std::mutex imuLock;
	std::mutex odoLock;
	// 发布当前帧校正后点云，有效点
	ros::Subscriber subLaserCloud;
	ros::Publisher pubLaserCloud;
	// imu数据队列(原始数据，转lidar系下)
	ros::Publisher pubExtractedCloud;
	ros::Publisher pubLaserCloudInfo;
	// imu数据队列(原始数据，转lidar系下)
	ros::Subscriber subImu;
	std::deque<sensor_msgs::Imu> imuQueue;
	// imu里程计队列
	ros::Subscriber subOdom;
	std::deque<nav_msgs::Odometry> odomQueue;
	// 队列front帧，作为当前处理帧点云
	std::deque<sensor_msgs::PointCloud2> cloudQueue;
	sensor_msgs::PointCloud2 currentCloudMsg;
	// 当前激光帧起止时刻间对应的imu数据，计算相对于起始时刻的旋转增量，以及时间戳；用于插值计算当前激光帧起止时间范围内，每一时刻的旋转姿态
	double *imuTime = new double[queueLength];
	double *imuRotX = new double[queueLength];
	double *imuRotY = new double[queueLength];
	double *imuRotZ = new double[queueLength];

	int imuPointerCur;
	bool firstPointFlag;
	Eigen::Affine3f transStartInverse;
	// 当前帧原始激光点云
	pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
	pcl::PointCloud<PointXYZIRT>::Ptr tmpVelodyneCloudIn;
	pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
	pcl::PointCloud<MulranPointXYZIRT>::Ptr tmpMulranCloudIn;
	// 当前帧运动畸变校正之后的激光点云
	pcl::PointCloud<PointType>::Ptr fullCloud;
	// 从fullcloud中提取的有效点
	pcl::PointCloud<PointType>::Ptr extractedCloud;

	int deskewFlag;
	cv::Mat rangeMat;

	bool odomDeskewFlag;
	// 当前激光帧起止时刻对应imu里程计位姿变换，该变换对应的平移增量；用于插值计算当前激光帧起止时间范围内，每一时刻的位置
	float odomIncreX;
	float odomIncreY;
	float odomIncreZ;
	// 当前帧激光点云运动畸变校正之后的数据，包括点云数据、初始位姿、姿态角等，发布给featureExatruction进行特征提取
	lio_sam::cloud_info cloudInfo;
	// 当前帧起始时刻
	double timeScanCur;
	// 当前帧结束时刻
	double timeScanEnd;
	// 当前帧header，包含时间戳信息
	std_msgs::Header cloudHeader;

public:
	// 构造函数
	ImageProjection() : deskewFlag(0)
	{
		// 订阅原始IMU数据
		subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
		// 订阅IMU里程计，由imuPreintegration积分计算得到的每时刻IMU位姿
		subOdom = nh.subscribe<nav_msgs::Odometry>(odomTopic + "_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
		// 订阅原始lidar数据
		subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
		// 发布当前激光帧运动畸变校正后的点云，有效点
		pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/deskew/cloud_deskewed", 1);
		// 发布当前激光帧运动畸变校正后的点云信息
		pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1);
		// 为变量申请内存
		allocateMemory();
		// 重置参数
		resetParameters();
		// pcl日志级别，只打开ERROR日志
		pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
	}
	// 初始化，为变量申请内存
	void allocateMemory()
	{
		laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
		tmpVelodyneCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
		tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
		tmpMulranCloudIn.reset(new pcl::PointCloud<MulranPointXYZIRT>());
		fullCloud.reset(new pcl::PointCloud<PointType>());
		extractedCloud.reset(new pcl::PointCloud<PointType>());

		fullCloud->points.resize(N_SCAN * Horizon_SCAN);

		cloudInfo.startRingIndex.assign(N_SCAN, 0);
		cloudInfo.endRingIndex.assign(N_SCAN, 0);

		cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
		cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

		resetParameters();
	}
	// 重置参数，接收每帧lidar数据都要重置这些参数
	void resetParameters()
	{
		laserCloudIn->clear();
		extractedCloud->clear();
		// reset range matrix for range image projection
		rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

		imuPointerCur = 0;
		firstPointFlag = true;
		odomDeskewFlag = false;

		for (int i = 0; i < queueLength; ++i)
		{
			imuTime[i] = 0;
			imuRotX[i] = 0;
			imuRotY[i] = 0;
			imuRotZ[i] = 0;
		}
	}

	~ImageProjection() {}
	/* IMU原始数据回调函数
			IMU原始测量数据转换到lidar系下，加速度、角速度、RPY姿态 */
	void imuHandler(const sensor_msgs::Imu::ConstPtr &imuMsg)
	{
		sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

		std::lock_guard<std::mutex> lock1(imuLock);
		imuQueue.push_back(thisImu);

		// debug IMU data
		// cout << std::setprecision(6);
		// cout << "IMU acc: " << endl;
		// cout << "x: " << thisImu.linear_acceleration.x <<
		//       ", y: " << thisImu.linear_acceleration.y <<
		//       ", z: " << thisImu.linear_acceleration.z << endl;
		// cout << "IMU gyro: " << endl;
		// cout << "x: " << thisImu.angular_velocity.x <<
		//       ", y: " << thisImu.angular_velocity.y <<
		//       ", z: " << thisImu.angular_velocity.z << endl;
		// double imuRoll, imuPitch, imuYaw;
		// tf::Quaternion orientation;
		// tf::quaternionMsgToTF(thisImu.orientation, orientation);
		// tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
		// cout << "IMU roll pitch yaw: " << endl;
		// cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
	}
	// 订阅IMU里程计，由imuPreintegration积分计算得到的每时刻imu位姿
	void odometryHandler(const nav_msgs::Odometry::ConstPtr &odometryMsg)
	{
		std::lock_guard<std::mutex> lock2(odoLock);
		odomQueue.push_back(*odometryMsg);
	}
	/**
	 * 订阅原始lidar数据
	 * 1、添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
	 * 2、当前帧起止时刻对应的imu数据、imu里程计数据处理
	 *   imu数据：
	 *   1) 遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
	 *   2) 用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
	 *   imu里程计数据：
	 *   1) 遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
	 *   2) 用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
	 * 3、当前帧激光点云运动畸变校正
	 *   1) 检查激光点距离、扫描线是否合规
	 *   2) 激光运动畸变校正，保存激光点
	 * 4、提取有效激光点，存extractedCloud
	 * 5、发布当前帧校正后点云，有效点
	 * 6、重置参数，接收每帧lidar数据都要重置这些参数
	 */
	void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
	{
		// 添加一帧激光点云到点云队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
		if (!cachePointCloud(laserCloudMsg))
			return;
		// 当前帧起止时刻对应的imu数据、imu里程计数据处理
		if (!deskewInfo())
			return;
		// 当前帧激光点云运动畸变校正
		// 1.检查激光点距离、扫描线是否合规
		// 2.激光运动畸变校正，保存激光点
		projectPointCloud();
		// 提取有效激光点，存extractedCloud
		cloudExtraction();
		// 发布当前帧校正后点云，有效点
		publishClouds();
		// 重置参数，接收每帧lidar数据都要重置这些参数
		resetParameters();
	}
	// 添加一帧激光点云到点云队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
	bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
	{
		// cache point cloud
		cloudQueue.push_back(*laserCloudMsg);
		if (cloudQueue.size() <= 2)
			return false;

		// 取出激光点云队列最早的一帧
		currentCloudMsg = std::move(cloudQueue.front());
		// get timestamp
		cloudHeader = currentCloudMsg.header;
		// 当前帧起始时刻
		timeScanCur = cloudHeader.stamp.toSec();
		cloudQueue.pop_front();

		if (sensor == SensorType::VELODYNE)
		{
			// 转换成pcl点云格式
			pcl::moveFromROSMsg(currentCloudMsg, *tmpVelodyneCloudIn);
			for (size_t i = 0; i < tmpVelodyneCloudIn->size(); i++)
			{
				auto &src = tmpVelodyneCloudIn->points[i];
				PointXYZIRT dst;
				if (std::isnan(src.x) || std::isnan(src.y) || std::isnan(src.z))
					continue;
				dst.x = src.x;
				dst.y = src.y;
				dst.z = src.z;
				dst.intensity = src.intensity;
				dst.ring = src.ring;
				dst.time = src.time;
				laserCloudIn->push_back(dst);
			}
		}
		else if (sensor == SensorType::OUSTER)
		{
			// Convert to Velodyne format
			pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
			laserCloudIn->is_dense = true;
			for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
			{
				auto &src = tmpOusterCloudIn->points[i];
				PointXYZIRT dst;
				if (std::isnan(src.x) || std::isnan(src.y) || std::isnan(src.z))
					continue;
				dst.x = src.x;
				dst.y = src.y;
				dst.z = src.z;
				dst.intensity = src.intensity;
				dst.ring = src.ring;
				dst.time = src.t * 1e-9f;
				laserCloudIn->push_back(dst);
			}
		}
		else if (sensor == SensorType::MULRAN)
		{
			// Convert to Velodyne format
			pcl::moveFromROSMsg(currentCloudMsg, *tmpMulranCloudIn);
			laserCloudIn->is_dense = true;
			for (size_t i = 0; i < tmpMulranCloudIn->size(); i++)
			{
				auto &src = tmpMulranCloudIn->points[i];
				PointXYZIRT dst;
				if (std::isnan(src.x) || std::isnan(src.y) || std::isnan(src.z))
					continue;
				dst.x = src.x;
				dst.y = src.y;
				dst.z = src.z;
				dst.intensity = src.intensity;
				dst.ring = src.ring;
				dst.time = float(src.t);
				laserCloudIn->push_back(dst);
			}
		}
		else
		{
			ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
			ros::shutdown();
		}

		// 当前帧结束时刻，注：点云中激光点的time记录相对于当前帧第一个激光点的时差，第一个点time=0
		timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

		// check dense flag
		if (laserCloudIn->is_dense == false)
		{
			ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
			ros::shutdown();
		}

		// check ring channel 检查是否存在ring通道，注意static，只检查一次
		static int ringFlag = 0;
		if (ringFlag == 0)
		{
			ringFlag = -1;
			for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
			{
				if (currentCloudMsg.fields[i].name == "ring")
				{
					ringFlag = 1;
					break;
				}
			}
			if (ringFlag == -1)
			{
				ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
				ros::shutdown();
			}
		}

		// check point time 检查是否存在time通道
		if (deskewFlag == 0)
		{
			deskewFlag = -1;
			for (auto &field : currentCloudMsg.fields)
			{
				if (field.name == "time" || field.name == "t")
				{
					deskewFlag = 1;
					break;
				}
			}
			if (deskewFlag == -1)
				ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
		}

		return true;
	}
	// 当前帧起止时刻对应的imu数据、imu里程计数据处理
	bool deskewInfo()
	{
		std::lock_guard<std::mutex> lock1(imuLock);
		std::lock_guard<std::mutex> lock2(odoLock);

		// make sure IMU data available for the scan
		if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
		{
			ROS_DEBUG("Waiting for IMU data ...");
			return false;
		}
		/* 当前帧对应imu数据处理
		1、遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
		2、用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
		注：imu数据都已经转换到lidar系下了 */
		imuDeskewInfo();
		/* 当前帧对应imu里程计处理
		1、遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
		2、用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
		注：imu数据都已经转换到lidar系下了 */
		odomDeskewInfo();

		return true;
	}

	void imuDeskewInfo()
	{
		cloudInfo.imuAvailable = false;

		while (!imuQueue.empty())
		{
			if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
				imuQueue.pop_front();
			else
				break;
		}

		if (imuQueue.empty())
			return;

		imuPointerCur = 0;

		for (int i = 0; i < (int)imuQueue.size(); ++i)
		{
			sensor_msgs::Imu thisImuMsg = imuQueue[i];
			double currentImuTime = thisImuMsg.header.stamp.toSec();

			// get roll, pitch, and yaw estimation for this scan
			if (currentImuTime <= timeScanCur)
				imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

			if (currentImuTime > timeScanEnd + 0.01)
				break;

			if (imuPointerCur == 0)
			{
				imuRotX[0] = 0;
				imuRotY[0] = 0;
				imuRotZ[0] = 0;
				imuTime[0] = currentImuTime;
				++imuPointerCur;
				continue;
			}

			// get angular velocity
			double angular_x, angular_y, angular_z;
			imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

			// integrate rotation
			double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
			imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
			imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
			imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
			imuTime[imuPointerCur] = currentImuTime;
			++imuPointerCur;
		}

		--imuPointerCur;

		if (imuPointerCur <= 0)
			return;

		cloudInfo.imuAvailable = true;
	}

	void odomDeskewInfo()
	{
		cloudInfo.odomAvailable = false;

		while (!odomQueue.empty())
		{
			if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
				odomQueue.pop_front();
			else
				break;
		}

		if (odomQueue.empty())
			return;

		if (odomQueue.front().header.stamp.toSec() > timeScanCur)
			return;

		// get start odometry at the beinning of the scan
		nav_msgs::Odometry startOdomMsg;

		for (int i = 0; i < (int)odomQueue.size(); ++i)
		{
			startOdomMsg = odomQueue[i];

			if (ROS_TIME(&startOdomMsg) < timeScanCur)
				continue;
			else
				break;
		}

		tf::Quaternion orientation;
		tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

		double roll, pitch, yaw;
		tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

		// Initial guess used in mapOptimization
		cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
		cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
		cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
		cloudInfo.initialGuessRoll = roll;
		cloudInfo.initialGuessPitch = pitch;
		cloudInfo.initialGuessYaw = yaw;

		cloudInfo.odomAvailable = true;

		// get end odometry at the end of the scan
		odomDeskewFlag = false;

		if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
			return;

		nav_msgs::Odometry endOdomMsg;

		for (int i = 0; i < (int)odomQueue.size(); ++i)
		{
			endOdomMsg = odomQueue[i];

			if (ROS_TIME(&endOdomMsg) < timeScanEnd)
				continue;
			else
				break;
		}

		if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
			return;

		Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

		tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
		tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
		Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

		Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

		float rollIncre, pitchIncre, yawIncre;
		pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

		odomDeskewFlag = true;
	}

	void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
	{
		*rotXCur = 0;
		*rotYCur = 0;
		*rotZCur = 0;

		int imuPointerFront = 0;
		while (imuPointerFront < imuPointerCur)
		{
			if (pointTime < imuTime[imuPointerFront])
				break;
			++imuPointerFront;
		}

		if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
		{
			*rotXCur = imuRotX[imuPointerFront];
			*rotYCur = imuRotY[imuPointerFront];
			*rotZCur = imuRotZ[imuPointerFront];
		}
		else
		{
			int imuPointerBack = imuPointerFront - 1;
			double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
			double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
			*rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
			*rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
			*rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
		}
	}

	void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
	{
		*posXCur = 0;
		*posYCur = 0;
		*posZCur = 0;

		// If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

		// if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
		//     return;

		// float ratio = relTime / (timeScanEnd - timeScanCur);

		// *posXCur = ratio * odomIncreX;
		// *posYCur = ratio * odomIncreY;
		// *posZCur = ratio * odomIncreZ;
	}

	PointType deskewPoint(PointType *point, double relTime)
	{
		if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
			return *point;

		double pointTime = timeScanCur + relTime;

		float rotXCur, rotYCur, rotZCur;
		findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

		float posXCur, posYCur, posZCur;
		findPosition(relTime, &posXCur, &posYCur, &posZCur);

		if (firstPointFlag == true)
		{
			transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
			firstPointFlag = false;
		}

		// transform points to start
		Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
		Eigen::Affine3f transBt = transStartInverse * transFinal;

		PointType newPoint;
		newPoint.x = transBt(0, 0) * point->x + transBt(0, 1) * point->y + transBt(0, 2) * point->z + transBt(0, 3);
		newPoint.y = transBt(1, 0) * point->x + transBt(1, 1) * point->y + transBt(1, 2) * point->z + transBt(1, 3);
		newPoint.z = transBt(2, 0) * point->x + transBt(2, 1) * point->y + transBt(2, 2) * point->z + transBt(2, 3);
		newPoint.intensity = point->intensity;

		return newPoint;
	}

	void projectPointCloud()
	{
		int cloudSize = laserCloudIn->points.size();
		// range image projection
		for (int i = 0; i < cloudSize; ++i)
		{
			PointType thisPoint;
			thisPoint.x = laserCloudIn->points[i].x;
			thisPoint.y = laserCloudIn->points[i].y;
			thisPoint.z = laserCloudIn->points[i].z;
			thisPoint.intensity = laserCloudIn->points[i].intensity;

			float range = pointDistance(thisPoint);
			if (range < lidarMinRange || range > lidarMaxRange)
				continue;

			int rowIdn = laserCloudIn->points[i].ring;
			// int rowIdn = (i % 64) + 1 ; // for MulRan dataset, Ouster OS1-64 .bin file,  giseop

			if (rowIdn < 0 || rowIdn >= N_SCAN)
				continue;

			if (rowIdn % downsampleRate != 0)
				continue;

			float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

			static float ang_res_x = 360.0 / float(Horizon_SCAN);
			int columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
			if (columnIdn >= Horizon_SCAN)
				columnIdn -= Horizon_SCAN;

			if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
				continue;

			if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
				continue;

			thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

			rangeMat.at<float>(rowIdn, columnIdn) = range;

			int index = columnIdn + rowIdn * Horizon_SCAN;
			fullCloud->points[index] = thisPoint;
		}
	}

	void cloudExtraction()
	{
		int count = 0;
		// extract segmented cloud for lidar odometry
		for (int i = 0; i < N_SCAN; ++i)
		{
			cloudInfo.startRingIndex[i] = count - 1 + 5;

			for (int j = 0; j < Horizon_SCAN; ++j)
			{
				if (rangeMat.at<float>(i, j) != FLT_MAX)
				{
					// mark the points' column index for marking occlusion later
					cloudInfo.pointColInd[count] = j;
					// save range info
					cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
					// save extracted cloud
					extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
					// size of extracted cloud
					++count;
				}
			}
			cloudInfo.endRingIndex[i] = count - 1 - 5;
		}
	}

	void publishClouds()
	{
		cloudInfo.header = cloudHeader;
		cloudInfo.cloud_deskewed = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
		pubLaserCloudInfo.publish(cloudInfo);
	}
};

int main(int argc, char **argv)
{
	ros::init(argc, argv, "lio_sam");

	ImageProjection IP;

	ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

	ros::MultiThreadedSpinner spinner(3);
	spinner.spin();

	return 0;
}
