#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
/*
 *订阅激光里程计（来自MapOptimization）和IMU里程计，
 *根据前一时刻激光里程计，和该时刻到当前时刻的IMU里程计变换增量，
 *计算当前时刻IMU里程计；rviz展示IMU里程计轨迹（局部）
 */
class TransformFusion : public ParamServer
{
public:
	std::mutex mtx;

	ros::Subscriber subImuOdometry;
	ros::Subscriber subLaserOdometry;

	ros::Publisher pubImuOdometry;
	ros::Publisher pubImuPath;

	Eigen::Affine3f lidarOdomAffine;
	Eigen::Affine3f imuOdomAffineFront;
	Eigen::Affine3f imuOdomAffineBack;

	tf::TransformListener tfListener;
	tf::StampedTransform lidar2Baselink;

	double lidarOdomTime = -1;
	deque<nav_msgs::Odometry> imuOdomQueue;

	TransformFusion()
	{
		// 如果lidar系与baselink系不同（激光系和载体系），需要外部提供二者之间的变换关系
		if (lidarFrame != baselinkFrame)
		{
			try
			{
				// 等待3s
				tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
				// lidar系到baselink系的变换
				tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
			}
			catch (tf::TransformException ex)
			{
				ROS_ERROR("%s", ex.what());
			}
		}
		// 订阅激光里程计，来自mapOptimization
		subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
		// 订阅imu里程计，来自IMUPreintegration
		subImuOdometry = nh.subscribe<nav_msgs::Odometry>(odomTopic + "_incremental", 2000, &TransformFusion::imuOdometryHandler, this, ros::TransportHints().tcpNoDelay());
		// 发布imu里程计，用于rviz展示
		pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
		// 发布imu里程计轨迹
		pubImuPath = nh.advertise<nav_msgs::Path>("lio_sam/imu/path", 1);
	}
	/**
	 * 里程计对应变换矩阵
	 */
	Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
	{
		double x, y, z, roll, pitch, yaw;
		x = odom.pose.pose.position.x;
		y = odom.pose.pose.position.y;
		z = odom.pose.pose.position.z;
		tf::Quaternion orientation;
		tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
		tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
		return pcl::getTransformation(x, y, z, roll, pitch, yaw);
	}
	/**
	 * 订阅激光里程计，来自mapOptimization
	 */
	void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
	{
		std::lock_guard<std::mutex> lock(mtx);
		// 激光里程计对应变换矩阵
		lidarOdomAffine = odom2affine(*odomMsg);
		// 激光里程计时间戳
		lidarOdomTime = odomMsg->header.stamp.toSec();
	}
	/**
	 * 订阅imu里程计，来自IMUPreintegration
	 * 1、以最近一帧激光里程计位姿为基础，计算该时刻与当前时刻间imu里程计增量位姿变换，相乘得到当前时刻imu里程计位姿
	 * 2、发布当前时刻里程计位姿，用于rviz展示；发布imu里程计路径，注：只是最近一帧激光里程计时刻与当前时刻之间的一段
	 */
	void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
	{
		// 发布tf，map与odom系设为同一个系
		static tf::TransformBroadcaster tfMap2Odom;
		static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
		tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

		std::lock_guard<std::mutex> lock(mtx);
		// 添加imu里程计到队列
		imuOdomQueue.push_back(*odomMsg);

		// get latest odometry (at current IMU stamp)
		// 从imu里程计队列中删除当前（最近的一帧）激光里程计时刻之前的数据
		if (lidarOdomTime == -1)
			return;
		while (!imuOdomQueue.empty())
		{
			if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
				imuOdomQueue.pop_front();
			else
				break;
		}
		// 最近的一帧激光里程计时刻对应imu里程计位姿
		Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
		// 当前时刻imu里程计位姿
		Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
		// imu里程计增量位姿变换
		Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
		// 最近的一帧激光里程计位姿 * imu里程计增量位姿变换 = 当前时刻imu里程计位姿
		Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
		float x, y, z, roll, pitch, yaw;
		pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);

		// publish latest odometry
		// 发布当前时刻里程计位姿
		nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
		laserOdometry.pose.pose.position.x = x;
		laserOdometry.pose.pose.position.y = y;
		laserOdometry.pose.pose.position.z = z;
		laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
		pubImuOdometry.publish(laserOdometry);

		// 发布tf，当前时刻odom与baselink系变换关系
		static tf::TransformBroadcaster tfOdom2BaseLink;
		tf::Transform tCur;
		tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
		if (lidarFrame != baselinkFrame)
			tCur = tCur * lidar2Baselink;
		tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
		tfOdom2BaseLink.sendTransform(odom_2_baselink);

		// 发布imu里程计路径，注：只是最近一帧激光里程计时刻与当前时刻之间的一段
		static nav_msgs::Path imuPath;
		static double last_path_time = -1;
		double imuTime = imuOdomQueue.back().header.stamp.toSec();
		// 每隔0.1s添加一个
		if (imuTime - last_path_time > 0.1)
		{
			last_path_time = imuTime;
			geometry_msgs::PoseStamped pose_stamped;
			pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
			pose_stamped.header.frame_id = odometryFrame;
			pose_stamped.pose = laserOdometry.pose.pose;
			imuPath.poses.push_back(pose_stamped);
			// 删除最近一帧激光里程计时刻之前的imu里程计
			while (!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
				imuPath.poses.erase(imuPath.poses.begin());
			if (pubImuPath.getNumSubscribers() != 0)
			{
				imuPath.header.stamp = imuOdomQueue.back().header.stamp;
				imuPath.header.frame_id = odometryFrame;
				pubImuPath.publish(imuPath);
			}
		}
	}
};

class IMUPreintegration : public ParamServer
{
public:
	std::mutex mtx;

	ros::Subscriber subImu;
	ros::Subscriber subOdometry;
	ros::Publisher pubImuOdometry;

	bool systemInitialized = false;
	// 噪声协方差
	gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
	gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
	gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
	gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
	gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
	gtsam::Vector noiseModelBetweenBias;

	// imu预积分器
	gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
	gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;
	// imu数据队列
	std::deque<sensor_msgs::Imu> imuQueOpt;
	std::deque<sensor_msgs::Imu> imuQueImu;
	// imu因子图优化过程中的状态变量
	gtsam::Pose3 prevPose_;
	gtsam::Vector3 prevVel_;
	gtsam::NavState prevState_;
	gtsam::imuBias::ConstantBias prevBias_;
	// imu状态
	gtsam::NavState prevStateOdom;
	gtsam::imuBias::ConstantBias prevBiasOdom;

	bool doneFirstOpt = false;
	double lastImuT_imu = -1;
	double lastImuT_opt = -1;
	// ISAM2优化器
	gtsam::ISAM2 optimizer;
	gtsam::NonlinearFactorGraph graphFactors;
	gtsam::Values graphValues;

	const double delta_t = 0;

	int key = 1;
	// T_bl: tramsform points from lidar frame to imu frame
	gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
	// T_lb: tramsform points from imu frame to lidar frame
	gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));
	// 构造函数
	IMUPreintegration()
	{
		// 订阅imu原始数据，用下面因子图优化的结果，施加两帧之间的imu预积分量，预测每一时刻（imu频率）的imu里程计
		subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());
		// 订阅激光里程计，来自mapOptimization，用两帧之间的imu预计分量构建因子图，优化当前帧位姿（这个位姿仅用于更新每时刻的imu里程计，以及下一次因子图优化）
		subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 5, &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());
		// 发布imu里程计
		pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic + "_incremental", 2000);
		// imu预积分的噪声协方差
		boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
		p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);							// acc white noise in continuous
		p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);									// gyro white noise in continuous
		p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);											// error committed in integrating position from velocities
		gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // assume zero initial bias

		// 噪声先验
		priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
		priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);																															 // m/s
		priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);																														 // 1e-2 ~ 1e-3 seems to be good

		// 激光里程计scan-to-map优化过程中发生退化，则选择一个较大的协方差
		correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
		correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());							 // rad,rad,rad,m, m, m
		noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
		// imu预积分器，用于预测每一时刻(imu频率)的imu里程计(转到lidar系了，与激光里程计同一个系)
		imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread

		// imu预积分器，用于因子图优化
		imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization
	}
	// 重置ISAM2优化器
	void resetOptimization()
	{
		gtsam::ISAM2Params optParameters;
		optParameters.relinearizeThreshold = 0.1;
		optParameters.relinearizeSkip = 1;
		optimizer = gtsam::ISAM2(optParameters);

		gtsam::NonlinearFactorGraph newGraphFactors;
		graphFactors = newGraphFactors;

		gtsam::Values NewGraphValues;
		graphValues = NewGraphValues;
	}
	// 重置参数
	void resetParams()
	{
		lastImuT_imu = -1;
		doneFirstOpt = false;
		systemInitialized = false;
	}
	/**
	 * 订阅激光里程计，来自mapOptimization
	 * 1、每隔100帧激光里程计，重置ISAM2优化器，添加里程计、速度、偏置先验因子，执行优化
	 * 2、计算前一帧激光里程计与当前帧激光里程计之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始状态估计，添加来自mapOptimization的当前帧位姿，进行因子图优化，更新当前帧状态
	 * 3、优化之后，执行重传播；优化更新了imu的偏置，用最新的偏置重新计算当前激光里程计时刻之后的imu预积分，这个预积分用于计算每时刻位姿
	 */
	void odometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
	{
		std::lock_guard<std::mutex> lock(mtx);

		double currentCorrectionTime = ROS_TIME(odomMsg);

		// make sure we have imu data to integrate
		if (imuQueOpt.empty())
			return;

		float p_x = odomMsg->pose.pose.position.x;
		float p_y = odomMsg->pose.pose.position.y;
		float p_z = odomMsg->pose.pose.position.z;
		float r_x = odomMsg->pose.pose.orientation.x;
		float r_y = odomMsg->pose.pose.orientation.y;
		float r_z = odomMsg->pose.pose.orientation.z;
		float r_w = odomMsg->pose.pose.orientation.w;
		bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
		gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

		// 0. initialize system
		if (systemInitialized == false)
		{
			resetOptimization();

			// pop old IMU message
			while (!imuQueOpt.empty())
			{
				if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
				{
					lastImuT_opt = ROS_TIME(&imuQueOpt.front());
					imuQueOpt.pop_front();
				}
				else
					break;
			}
			// initial pose
			prevPose_ = lidarPose.compose(lidar2Imu);
			gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
			graphFactors.add(priorPose);
			// initial velocity
			prevVel_ = gtsam::Vector3(0, 0, 0);
			gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
			graphFactors.add(priorVel);
			// initial bias
			prevBias_ = gtsam::imuBias::ConstantBias();
			gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
			graphFactors.add(priorBias);
			// add values
			graphValues.insert(X(0), prevPose_);
			graphValues.insert(V(0), prevVel_);
			graphValues.insert(B(0), prevBias_);
			// optimize once
			optimizer.update(graphFactors, graphValues);
			graphFactors.resize(0);
			graphValues.clear();

			imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
			imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

			key = 1;
			systemInitialized = true;
			return;
		}

		// reset graph for speed
		if (key == 100)
		{
			// get updated noise before reset
			gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
			gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
			gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
			// reset graph
			resetOptimization();
			// add pose
			gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
			graphFactors.add(priorPose);
			// add velocity
			gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
			graphFactors.add(priorVel);
			// add bias
			gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
			graphFactors.add(priorBias);
			// add values
			graphValues.insert(X(0), prevPose_);
			graphValues.insert(V(0), prevVel_);
			graphValues.insert(B(0), prevBias_);
			// optimize once
			optimizer.update(graphFactors, graphValues);
			graphFactors.resize(0);
			graphValues.clear();

			key = 1;
		}

		// 1. integrate imu data and optimize
		while (!imuQueOpt.empty())
		{
			// pop and integrate imu data that is between two optimizations
			sensor_msgs::Imu *thisImu = &imuQueOpt.front();
			double imuTime = ROS_TIME(thisImu);
			if (imuTime < currentCorrectionTime - delta_t)
			{
				double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
				imuIntegratorOpt_->integrateMeasurement(
						gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
						gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);

				lastImuT_opt = imuTime;
				imuQueOpt.pop_front();
			}
			else
				break;
		}
		// add imu factor to graph
		const gtsam::PreintegratedImuMeasurements &preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
		gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
		graphFactors.add(imu_factor);
		// add imu bias between factor
		graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
																																				gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
		// add pose factor
		gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
		gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
		graphFactors.add(pose_factor);
		// insert predicted values
		gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
		graphValues.insert(X(key), propState_.pose());
		graphValues.insert(V(key), propState_.v());
		graphValues.insert(B(key), prevBias_);
		// optimize
		optimizer.update(graphFactors, graphValues);
		optimizer.update();
		graphFactors.resize(0);
		graphValues.clear();
		// Overwrite the beginning of the preintegration for the next step.
		gtsam::Values result = optimizer.calculateEstimate();
		prevPose_ = result.at<gtsam::Pose3>(X(key));
		prevVel_ = result.at<gtsam::Vector3>(V(key));
		prevState_ = gtsam::NavState(prevPose_, prevVel_);
		prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
		// Reset the optimization preintegration object.
		imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
		// check optimization
		if (failureDetection(prevVel_, prevBias_))
		{
			resetParams();
			return;
		}

		// 2. after optiization, re-propagate imu odometry preintegration
		prevStateOdom = prevState_;
		prevBiasOdom = prevBias_;
		// first pop imu message older than current correction data
		double lastImuQT = -1;
		while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
		{
			lastImuQT = ROS_TIME(&imuQueImu.front());
			imuQueImu.pop_front();
		}
		// repropogate
		if (!imuQueImu.empty())
		{
			// reset bias use the newly optimized bias
			imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
			// integrate imu message from the beginning of this optimization
			for (int i = 0; i < (int)imuQueImu.size(); ++i)
			{
				sensor_msgs::Imu *thisImu = &imuQueImu[i];
				double imuTime = ROS_TIME(thisImu);
				double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);

				imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
																								gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);
				lastImuQT = imuTime;
			}
		}

		++key;
		doneFirstOpt = true;
	}

	bool failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur)
	{
		Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
		if (vel.norm() > 30)
		{
			ROS_WARN("Large velocity, reset IMU-preintegration!");
			return true;
		}

		Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
		Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
		if (ba.norm() > 1.0 || bg.norm() > 1.0)
		{
			ROS_WARN("Large bias, reset IMU-preintegration!");
			return true;
		}

		return false;
	}
	/**
	 * 订阅imu原始数据
	 * 1、用上一帧激光里程计时刻对应的状态、偏置，施加从该时刻开始到当前时刻的imu预积分量，得到当前时刻的状态，也就是imu里程计
	 * 2、imu里程计位姿转到lidar系，发布里程计
	 */
	void imuHandler(const sensor_msgs::Imu::ConstPtr &imu_raw)
	{
		std::lock_guard<std::mutex> lock(mtx);
		// 将imu原始测量数据转换到雷达坐标系下，加速度、角速度和姿态信息
		sensor_msgs::Imu thisImu = imuConverter(*imu_raw);
		// 将IMU信息存到两个队列里，一个用于优化，一个用于重积分
		imuQueOpt.push_back(thisImu);
		imuQueImu.push_back(thisImu);

		if (doneFirstOpt == false)
			return;

		double imuTime = ROS_TIME(&thisImu);																			 // 当前帧imu时间戳
		double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu); // 获取相邻两帧imu数据时间差
		lastImuT_imu = imuTime;

		// integrate this single imu message
		// 记录imu的测量信息
		// 此时用的imu预积分器为imuIntegeratorImu_
		imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
																						gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z), dt);

		// predict odometry
		// 利用上一时刻的imu里程计状态信息PVQ和偏置信息，预积分出当前时刻imu里程计状态信息PVQ
		gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

		// publish odometry
		nav_msgs::Odometry odometry;
		odometry.header.stamp = thisImu.header.stamp;
		odometry.header.frame_id = odometryFrame;
		odometry.child_frame_id = "odom_imu";

		// transform imu pose to ldiar
		gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
		gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

		odometry.pose.pose.position.x = lidarPose.translation().x();
		odometry.pose.pose.position.y = lidarPose.translation().y();
		odometry.pose.pose.position.z = lidarPose.translation().z();
		odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
		odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
		odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
		odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

		odometry.twist.twist.linear.x = currentState.velocity().x();
		odometry.twist.twist.linear.y = currentState.velocity().y();
		odometry.twist.twist.linear.z = currentState.velocity().z();
		odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
		odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
		odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
		pubImuOdometry.publish(odometry);
	}
};

int main(int argc, char **argv)
{
	ros::init(argc, argv, "roboat_loam");

	IMUPreintegration ImuP;

	TransformFusion TF;

	ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");

	ros::MultiThreadedSpinner spinner(4);
	spinner.spin();

	return 0;
}
