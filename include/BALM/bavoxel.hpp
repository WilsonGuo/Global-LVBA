#ifndef BAVOXEL_HPP
#define BAVOXEL_HPP

#include "tools.hpp"
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <thread>

#define WIN_SIZE 10
#define AVG_THR

const int layer_limit = 2;
// const int layer_size[] = {30, 30, 30, 30};
// float eigen_value_array[] = {1.0/4.0, 1.0/4.0, 1.0/4.0};
// float eigen_value_array[4] = {1.0/16, 1.0/16, 1.0/16, 1.0/16};
inline float eigen_ratio_array[4] = {0.3, 0.1, 0.06, 0.03};

inline void set_eigen_ratio_array(const std::array<float, 4>& arr)
{
  std::copy(arr.begin(), arr.end(), eigen_ratio_array);
}

const int min_ps = 15;
const int thd_num = 16;
const double one_three = (1.0 / 3.0);

// const double voxel_size = 1;
// const int life_span = 1000;
// const int merge_enable = 1;

class VOX_HESS
{
public:
  vector<const vector<PointCluster> *> plvec_voxels;
  int win_size;

  VOX_HESS(int _win_size = WIN_SIZE) : win_size(_win_size) {}

  ~VOX_HESS()
  {
    vector<const vector<PointCluster> *>().swap(plvec_voxels);
  }

  void push_voxel(const vector<PointCluster> *sig_orig, const vector<PLV(3)> *vec_orig)
  {
    int process_size = 0;
    for(int i = 0; i < win_size; i++)
      if((*sig_orig)[i].N != 0)
        process_size++;

    if(process_size < 2) return;
    plvec_voxels.push_back(sig_orig);
  }

  Eigen::Matrix<double, 6, 1> lam_f(Eigen::Vector3d *u, int m, int n)
  {
      Eigen::Matrix<double, 6, 1> jac;
      jac[0] = u[m][0] * u[n][0];
      jac[1] = u[m][0] * u[n][1] + u[m][1] * u[n][0];
      jac[2] = u[m][0] * u[n][2] + u[m][2] * u[n][0];
      jac[3] = u[m][1] * u[n][1];
      jac[4] = u[m][1] * u[n][2] + u[m][2] * u[n][1];
      jac[5] = u[m][2] * u[n][2];
      return jac;
  }

  void acc_evaluate2(const vector<IMUST> &xs, int head, int end,
                      Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    vector<PointCluster> sig_tran(win_size);
    const int kk = 0;

    PLV(3)
    viRiTuk(win_size);
    PLM(3)
    viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for(int a = head; a < end; a++)
    {
      const vector<PointCluster>& sig_orig = *plvec_voxels[a];
      vector<PointCluster> sig_tran(win_size);
      PointCluster sig;
      for(int i = 0; i < win_size; i++)
        if(sig_orig[i].N != 0)
        {
          sig_tran[i].transform(sig_orig[i], xs[i]);
          sig += sig_tran[i];
        }
      
      const Eigen::Vector3d& vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      const Eigen::Vector3d& lmbd = saes.eigenvalues();
      const Eigen::Matrix3d& U = saes.eigenvectors();
      int NN = sig.N;
      
      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

      const Eigen::Vector3d& uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for(int i = 0; i < 3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for(int i = 0; i < win_size; i++)
        if(sig_orig[i].N != 0)
        {
          Eigen::Matrix3d Pi = sig_orig[i].P;
          Eigen::Vector3d vi = sig_orig[i].v;
          Eigen::Matrix3d Ri = xs[i].R;
          double ni = sig_orig[i].N;

          Eigen::Matrix3d vihat; vihat << SKEW_SYM_MATRX(vi);
          Eigen::Vector3d RiTuk = Ri.transpose() * uk;
          Eigen::Matrix3d RiTukhat; RiTukhat << SKEW_SYM_MATRX(RiTuk);

          Eigen::Vector3d PiRiTuk = Pi * RiTuk;
          viRiTuk[i] = vihat * RiTuk;
          viRiTukukT[i] = viRiTuk[i] * uk.transpose();
          
          Eigen::Vector3d ti_v = xs[i].p - vBar;
          double ukTti_v = uk.dot(ti_v);

          Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
          Eigen::Vector3d combo2 = Ri*vi + ni*ti_v;
          Auk[i].block<3, 3>(0, 0) = (Ri*Pi + ti_v*vi.transpose()) * RiTukhat - Ri*combo1;
          Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
          Auk[i] /= NN;

          const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
          JacT.block<6, 1>(6*i, 0) += jjt;

          const Eigen::Matrix3d &HRt = 2.0/NN * (1.0-ni/NN) * viRiTukukT[i];
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
          Hb.block<3, 3>(0, 0) +=
            2.0/NN*(combo1-RiTukhat*Pi)*RiTukhat - 2.0/NN/NN*viRiTuk[i]*viRiTuk[i].transpose() - 0.5*hat(jjt.block<3, 1>(0, 0));
          Hb.block<3, 3>(0, 3) += HRt;
          Hb.block<3, 3>(3, 0) += HRt.transpose();
          Hb.block<3, 3>(3, 3) += 2.0/NN * (ni - ni*ni/NN) * ukukT;

          Hess.block<6, 6>(6*i, 6*i) += Hb;
        }
      
      for(int i = 0; i < win_size-1; i++)
        if(sig_orig[i].N != 0)
        {
          double ni = sig_orig[i].N;
          for(int j = i+1; j < win_size; j++)
            if(sig_orig[j].N != 0)
            {
              double nj = sig_orig[j].N;
              Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
              Hb.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTuk[i] * viRiTuk[j].transpose();
              Hb.block<3, 3>(0, 3) += -2.0*nj/NN/NN * viRiTukukT[i];
              Hb.block<3, 3>(3, 0) += -2.0*ni/NN/NN * viRiTukukT[j].transpose();
              Hb.block<3, 3>(3, 3) += -2.0*ni*nj/NN/NN * ukukT;

              Hess.block<6, 6>(6*i, 6*j) += Hb;
            }
        }
      residual += lmbd[kk];
    }

    for(int i = 1; i < win_size; i++)
      for(int j = 0; j < i; j++)
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
  }

  void evaluate_only_residual(const vector<IMUST> &xs, double &residual)
  {
    residual = 0;
    vector<PointCluster> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value

    int gps_size = plvec_voxels.size();

    for(int a = 0; a < gps_size; a++)
    {
      const vector<PointCluster>& sig_orig = *plvec_voxels[a];
      PointCluster sig;

      for(int i = 0; i < win_size; i++)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P/sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residual += lmbd[kk];
    }
  }

  // std::vector<double> evaluate_residual(const vector<IMUST> &xs)
  // {
  //     /* for outlier removal usage */
  //     std::vector<double> residuals;
  //     vector<PointCluster> sig_tran(win_size);
  //     int kk = 0; // The kk-th lambda value
  //     int gps_size = plvec_voxels.size();

  //     for(int a = 0; a < gps_size; a++)
  //     {
  //         const vector<PointCluster> &sig_orig = *plvec_voxels[a];
  //         PointCluster sig;

  //         for(int i = 0; i < win_size; i++)
  //         {
  //             sig_tran[i].transform(sig_orig[i], xs[i]);
  //             sig += sig_tran[i];
  //         }

  //         Eigen::Vector3d vBar = sig.v / sig.N;
  //         Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

  //         Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
  //         Eigen::Vector3d lmbd = saes.eigenvalues();

  //         residuals.push_back(lmbd[kk]);
  //     }

  //     return residuals;
  // }

  // void remove_residual(const vector<IMUST> &xs, double threshold, double reject_num)
  // {
  //     vector<PointCluster> sig_tran(win_size);
  //     int kk = 0; // The kk-th lambda value
  //     int rej_cnt = 0;
  //     size_t i = 0;
  //     for(; i < plvec_voxels.size();)
  //     {
  //         const vector<PointCluster> &sig_orig = *plvec_voxels[i];
  //         PointCluster sig;

  //         for(int j = 0; j < win_size; j++)
  //         {
  //             sig_tran[j].transform(sig_orig[j], xs[j]);
  //             sig += sig_tran[j];
  //         }

  //         Eigen::Vector3d vBar = sig.v / sig.N;
  //         Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();
  //         Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
  //         Eigen::Vector3d lmbd = saes.eigenvalues();

  //         if(lmbd[kk] >= threshold)
  //         {
  //             plvec_voxels.erase(plvec_voxels.begin() + i);
  //             rej_cnt++;
  //             continue;
  //         }
  //         i++;
  //         if(rej_cnt == reject_num)
  //             break;
  //     }
  // }
};

// int BINGO_CNT = 0;
enum OCTO_STATE
{
  UNKNOWN,
  MID_NODE,
  PLANE
};
class OCTO_TREE_NODE
{
public:
  OCTO_STATE octo_state;
  int layer, win_size;
  vector<PLV(3)> vec_orig;
  vector<PointCluster> sig_orig;

  OCTO_TREE_NODE *leaves[8];
  float voxel_center[3];
  float quater_length;
  float eigen_thr;

  Eigen::Vector3d center, direct, value_vector;
  double eigen_ratio;

  double ref;

#ifdef ENABLE_RVIZ
  ros::NodeHandle nh;
  ros::Publisher pub_residual = nh.advertise<sensor_msgs::PointCloud2>("/residual", 1000);
  ros::Publisher pub_direct = nh.advertise<visualization_msgs::MarkerArray>("/direct", 1000);
#endif

  OCTO_TREE_NODE(int _win_size = WIN_SIZE, float _eigen_thr = 1.0 / 10) : win_size(_win_size), eigen_thr(_eigen_thr)
  {
    ref = 255.0*rand()/(RAND_MAX + 1.0f);
    octo_state = UNKNOWN;
    layer = 0;
    vec_orig.resize(win_size);
    sig_orig.resize(win_size);
    for(int i = 0; i < 8; i++)
      leaves[i] = nullptr;
  }

  virtual ~OCTO_TREE_NODE()
  {
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        delete leaves[i];
  }

  OCTO_TREE_NODE* findCorrespondPoint(Eigen::Vector3d pw)
  {
    if ( octo_state == PLANE || (layer >= layer_limit)) return this;

    int xyz[3] = {0, 0, 0};
    xyz[0] = pw[0] > voxel_center[0] ? 1 : 0;
    xyz[1] = pw[1] > voxel_center[1] ? 1 : 0;
    xyz[2] = pw[2] > voxel_center[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

    // printf("leafnum: %d. \n", leafnum);

    return (leaves[leafnum] != nullptr) ? leaves[leafnum]->findCorrespondPoint(pw) : this;
  }

  bool judge_eigen(std::vector<IMUST>& x_buf)
  {
    PointCluster covMat;
    for(int i = 0; i < win_size; i++)
      if(sig_orig[i].N > 0)
      {
        PointCluster tmp;
        tmp.transform(sig_orig[i], x_buf[i]);
        covMat += tmp;
      }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
    value_vector = saes.eigenvalues();
    center = covMat.v / covMat.N;
    direct = saes.eigenvectors().col(0);

    eigen_ratio = saes.eigenvalues()[0] / saes.eigenvalues()[2]; // [0] is the smallest
    if(eigen_ratio > eigen_ratio_array[layer]) return 0;

    // double eva0 = saes.eigenvalues()[0];
    // double sqr_eva0 = sqrt(eva0);
    // Eigen::Vector3d center_turb = center + 5 * sqr_eva0 * direct;
    // vector<PointCluster> covMats(8);
    // for(int i = 0; i < win_size; i++)
    // {
    //   for(Eigen::Vector3d ap : vec_orig[i])
    //   {
    //     Eigen::Vector3d pvec = x_buf[i].R * ap + x_buf[i].p;
    //     int xyz[3] = {0, 0, 0};
    //     for(int k = 0; k < 3; k++)
    //       if(pvec(k) > center_turb[k])
    //         xyz[k] = 1;

    //     int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    //     covMats[leafnum].push(pvec);
    //   }
    // }

    // double ratios[2] = {1.0 / (3.0 * 3.0), 2.0 * 2.0};
    // int num_all = 0, num_qua = 0;
    // for(int i = 0; i < 8; i++)
    //   if(covMats[i].N > 10)
    //   {
    //     Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMats[i].cov());
    //     double child_eva0 = (saes.eigenvalues()[0]);
    //     if(child_eva0 > ratios[0] * eva0 && child_eva0 < ratios[1] * eva0)
    //       num_qua++;
    //     num_all++;
    //   }

    // double prop = 1.0 * num_qua / num_all;

    // if(prop < 0.5) return 0;
    return 1;
  }

  void cut_func(IMUST&x_buf, int ci)
  {
    PLV(3) &pvec_orig = vec_orig[ci];

    uint a_size = pvec_orig.size();
    for(uint j = 0; j < a_size; j++)
    {
      Eigen::Vector3d pvec_tran = x_buf.R * pvec_orig[j] + x_buf.p;
      int xyz[3] = {0, 0, 0};
      for(uint k = 0; k < 3; k++)
        if(pvec_tran[k] > voxel_center[k])
          xyz[k] = 1;
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OCTO_TREE_NODE(win_size, eigen_thr);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
        leaves[leafnum]->quater_length = quater_length / 2.0;
        leaves[leafnum]->layer = layer + 1;
      }

      leaves[leafnum]->vec_orig[ci].push_back(pvec_orig[j]);
      leaves[leafnum]->sig_orig[ci].push(pvec_orig[j]);
    }
    PLV(3)().swap(pvec_orig);
  }

  void recut(std::vector<IMUST>& x_buf)
  {
    if(octo_state == UNKNOWN)
    {
      int point_size = 0;
      for(int i = 0; i < win_size; i++)
        point_size += sig_orig[i].N;

      if(point_size < min_ps)
      {
        octo_state = MID_NODE;
        vector<PLV(3)>().swap(vec_orig);
        vector<PointCluster>().swap(sig_orig);
        return;
      }

      if(judge_eigen(x_buf))
      {
        octo_state = PLANE;
#ifndef ENABLE_FILTER
#ifndef ENABLE_RVIZ
        vector<PLV(3)>().swap(vec_orig);
#endif
#endif
        return;
      }
      else
      {
        if(layer == layer_limit)
        {
          octo_state = MID_NODE;
          vector<PLV(3)>().swap(vec_orig);
          vector<PointCluster>().swap(sig_orig);
          return;
        }
        vector<PointCluster>().swap(sig_orig);
        for(int i = 0; i < win_size; i++)
          cut_func(x_buf[i], i);
      }
    }

    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut(x_buf);
  }

  void tras_opt(VOX_HESS &vox_opt)
  {
    if(octo_state == PLANE)
      vox_opt.push_voxel(&sig_orig, &vec_orig);
    else
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt);
  }

  void tras_display(pcl::PointCloud<PointType> &pl_feat, std::vector<IMUST>& x_buf, int layer = 0)
  {
      // float ref = 255.0 * rand() / (RAND_MAX + 1.0f);
      // PointType ap;
      // ap.intensity = ref;

      // int tsize = 0;
      // for(int i=0; i< win_size; i++)
      //   tsize += vec_orig[i].size();
      // if(tsize < 100) return;

      if(octo_state == PLANE)
      {
          // std::vector<unsigned int> colors;
          // colors.push_back(static_cast<unsigned int>(rand() % 256));
          // colors.push_back(static_cast<unsigned int>(rand() % 256));
          // colors.push_back(static_cast<unsigned int>(rand() % 256));
          // pcl::PointCloud<pcl::PointXYZINormal> color_cloud;
          for(int i = 0; i < vec_orig.size(); i++)
          {
              for(size_t j = 0; j < vec_orig[i].size(); j++)
              {
                  Eigen::Vector3d pvec = x_buf[i].R * vec_orig[i][j] + x_buf[i].p;
                  
                  PointType ap;
                  ap.intensity = ref;
                  ap.x = pvec.x();
                  ap.y = pvec.y();
                  ap.z = pvec.z();
                  // ap.b = colors[0];
                  // ap.g = colors[1];
                  // ap.r = colors[2];
                  // ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
                  // ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
                  // ap.normal_z = sqrt(value_vector[0]);
                  // ap.curvature = total;
                  pl_feat.push_back(ap);
              }
          }

#ifdef ENABLE_RVIZ
          sensor_msgs::PointCloud2 dbg_msg;
          pcl::toROSMsg(color_cloud, dbg_msg);
          dbg_msg.header.frame_id = "camera_init";
          pub_residual.publish(dbg_msg);

          visualization_msgs::Marker marker;
          visualization_msgs::MarkerArray marker_array;
          marker.header.frame_id = "camera_init";
          marker.header.stamp = ros::Time::now();
          marker.ns = "basic_shapes";
          marker.id = BINGO_CNT;
          BINGO_CNT++;
          marker.action = visualization_msgs::Marker::ADD;
          marker.type = visualization_msgs::Marker::ARROW;
          marker.color.a = 1;
          marker.color.r = layer == 0 ? 1 : 0;
          marker.color.g = layer == 1 ? 1 : 0;
          marker.color.b = layer == 2 ? 1 : 0;
          marker.scale.x = 0.01;
          marker.scale.y = 0.05;
          marker.scale.z = 0.05;
          marker.lifetime = ros::Duration();
          geometry_msgs::Point apoint;
          apoint.x = center(0);
          apoint.y = center(1);
          apoint.z = center(2);
          marker.points.push_back(apoint);
          apoint.x += 0.2 * direct(0);
          apoint.y += 0.2 * direct(1);
          apoint.z += 0.2 * direct(2);
          marker.points.push_back(apoint);
          marker_array.markers.push_back(marker);
          pub_direct.publish(marker_array);
#endif
      }
      else
      {
          if(layer == layer_limit)
              return;
          layer++;
          for(int i = 0; i < 8; i++)
              if(leaves[i] != nullptr)
                  leaves[i]->tras_display(pl_feat, x_buf, layer);
      }
  }
};

class OCTO_TREE_ROOT : public OCTO_TREE_NODE
{
public:
    OCTO_TREE_ROOT(int _winsize, float _eigen_thr) : OCTO_TREE_NODE(_winsize, _eigen_thr) {}
};

bool iter_stop(Eigen::VectorXd &dx, double thre = 1e-7, int win_size = 0)
{
  // int win_size = dx.rows() / 6;
  if(win_size == 0)
    win_size = dx.rows() / 6;

  double angErr = 0, tranErr = 0;
  for(int i=0; i<win_size; i++)
  {
    angErr += dx.block<3, 1>(6*i, 0).norm();
    tranErr += dx.block<3, 1>(6*i+3, 0).norm();
  }

  angErr /= win_size; tranErr /= win_size;
  return (angErr < thre) && (tranErr < thre);
}

class BALM2
{
public:
    int win_size, jac_leng, imu_leng;
    BALM2(int _win_size = WIN_SIZE) : win_size(_win_size)
    {
        jac_leng = DVEL * win_size;
        imu_leng = DIM * win_size;
    }

    double divide_thread(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab,
                         Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
    {
        double residual = 0;
        Hess.setZero();
        JacT.setZero();
        PLM(-1)
        hessians(thd_num);
        PLV(-1)
        jacobins(thd_num);

        for(int i = 0; i < thd_num; i++)
        {
            hessians[i].resize(jac_leng, jac_leng);
            jacobins[i].resize(jac_leng);
        }

        int tthd_num = thd_num;
        vector<double> resis(tthd_num, 0);
        int g_size = voxhess.plvec_voxels.size();
        if(g_size < tthd_num)
            tthd_num = 1;

        vector<thread *> mthreads(tthd_num);
        double part = 1.0 * g_size / tthd_num;
        for(int i = 0; i < tthd_num; i++)
            mthreads[i] = new thread(&VOX_HESS::acc_evaluate2, &voxhess, x_stats, part * i, part * (i + 1),
                                     ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

        for(int i = 0; i < tthd_num; i++)
        {
            mthreads[i]->join();
            Hess += hessians[i];
            JacT += jacobins[i];
            residual += resis[i];
            delete mthreads[i];
        }
#ifdef AVG_THR
        return residual / g_size;
#else
        return residual;
#endif
    }

    double only_residual(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab, bool is_avg = false)
    {
        double residual2 = 0;
        voxhess.evaluate_only_residual(x_stats, residual2);
        if(is_avg)
            return residual2 / voxhess.plvec_voxels.size();
        return residual2;
    }

    // void remove_outlier(vector<IMUST> &x_stats, VOX_HESS &voxhess, double ratio)
    // {
    //     std::vector<double> residuals = voxhess.evaluate_residual(x_stats);
    //     std::sort(residuals.begin(), residuals.end()); // sort in ascending order
    //     double threshold = residuals[std::floor((1 - ratio) * voxhess.plvec_voxels.size()) - 1];
    //     int reject_num = std::floor(ratio * voxhess.plvec_voxels.size());
    //     // std::cout << "vox_num before " << voxhess.plvec_voxels.size();
    //     // std::cout << ", reject threshold " << std::setprecision(3) << threshold << ", rejected " << reject_num;
    //     voxhess.remove_residual(x_stats, threshold, reject_num);
    //     // std::cout << ", vox_num after " << voxhess.plvec_voxels.size() << std::endl;
    // }

    void damping_iter(vector<IMUST> &x_stats, VOX_HESS &voxhess)
    {
        double u = 0.01, v = 2;
        Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng),
            HessuD(jac_leng, jac_leng);
        Eigen::VectorXd JacT(jac_leng), dxi(jac_leng), new_dxi(jac_leng);

        D.setIdentity();
        double residual1, residual2, q;
        bool is_calc_hess = true;
        vector<IMUST> x_stats_temp;

        vector<IMUST> x_ab(win_size);
        x_ab[0] = x_stats[0];
        for(int i = 1; i < win_size; i++)
        {
            x_ab[i].p = x_stats[i - 1].R.transpose() * (x_stats[i].p - x_stats[i - 1].p);
            x_ab[i].R = x_stats[i - 1].R.transpose() * x_stats[i].R;
        }

        double hesstime = 0;
        double solvtime = 0;
        size_t max_mem = 0;
        // double loop_num = 0;
        for(int i = 0; i < 10; i++)
        {
            if(is_calc_hess)
                residual1 = divide_thread(x_stats, voxhess, x_ab, Hess, JacT);

            // double tm = ros::Time::now().toSec();
            D.diagonal() = Hess.diagonal();
            HessuD = Hess + u * D;
            // double t1 = ros::Time::now().toSec();
            Eigen::SparseMatrix<double> A1_sparse(jac_leng, jac_leng);
            std::vector<Eigen::Triplet<double>> tripletlist;
            for(int a = 0; a < jac_leng; a++)
                for(int b = 0; b < jac_leng; b++)
                    if(HessuD(a, b) != 0)
                    {
                        tripletlist.push_back(Eigen::Triplet<double>(a, b, HessuD(a, b)));
                        // A1_sparse.insert(a, b) = HessuD(a, b);
                    }
            A1_sparse.setFromTriplets(tripletlist.begin(), tripletlist.end());
            A1_sparse.makeCompressed();
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver_sparse;
            Solver_sparse.compute(A1_sparse);
            // size_t temp_mem = check_mem();
            // if(temp_mem > max_mem) max_mem = temp_mem;
            dxi = Solver_sparse.solve(-JacT);
            // temp_mem = check_mem();
            // if(temp_mem > max_mem) max_mem = temp_mem;
            // solvtime += ros::Time::now().toSec() - tm;
            // new_dxi = Solver_sparse.solve(-JacT);
            // printf("new solve time cost %f\n",ros::Time::now().toSec() - t1);
            // relative_err = ((Hess + u*D)*dxi + JacT).norm()/JacT.norm();
            // absolute_err = ((Hess + u*D)*dxi + JacT).norm();
            // std::cout<<"relative error "<<relative_err<<std::endl;
            // std::cout<<"absolute error "<<absolute_err<<std::endl;
            // std::cout<<"delta x\n"<<(new_dxi-dxi).transpose()/dxi.norm()<<std::endl;

            x_stats_temp = x_stats;
            for(int j = 0; j < win_size; j++)
            {
                x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL * j, 0));
                x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL * j + 3, 0);
            }

            double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);
#ifdef AVG_THR
            residual2 = only_residual(x_stats_temp, voxhess, x_ab, true);
            q1 /= voxhess.plvec_voxels.size();
#else
            residual2 = only_residual(x_stats_temp, voxhess, x_ab);
#endif
            q = (residual1 - residual2);
            // printf("iter%d: (%lf %lf) u: %lf v: %lf q: %lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

            // loop_num = i + 1;
            // if(hesstime/loop_num > 1) printf("Avg. Hessian time: %lf ", hesstime/loop_num);
            // if(solvtime/loop_num > 1) printf("Avg. solve time: %lf\n", solvtime/loop_num);
            // if(double(max_mem/1048576.0) > 2.0) printf("Max mem: %lf\n", double(max_mem/1048576.0));

            if(q > 0)
            {
                x_stats = x_stats_temp;
                q = q / q1;
                v = 2;
                q = 1 - pow(2 * q - 1, 3);
                u *= (q < one_three ? one_three : q);
                is_calc_hess = true;
            }
            else
            {
                u = u * v;
                v = 2 * v;
                is_calc_hess = false;
            }
#ifdef AVG_THR
            if((fabs(residual1 - residual2) / residual1) < 1e-6) //0.05
                break;
#else
            if(fabs(residual1 - residual2) < 1e-9)
                break;
#endif
        }
    }

    size_t check_mem()
    {
        FILE *file = fopen("/proc/self/status", "r");
        int result = -1;
        char line[128];

        while (fgets(line, 128, file) != nullptr)
        {
            if(strncmp(line, "VmRSS:", 6) == 0)
            {
                int len = strlen(line);

                const char *p = line;
                for(; std::isdigit(*p) == false; ++p)
                {
                }

                line[len - 3] = 0;
                result = atoi(p);

                break;
            }
        }
        fclose(file);

        return result;
    }
};


void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> &feat_map, pcl::PointCloud<PointType> &pl_feat,
               const IMUST &x_key, int fnum, int win_size,
               double voxel_size = 1.0, float eigen_ratio = 0.1)
{
  float loc_xyz[3];
  for(PointType &p_c : pl_feat.points)
  {
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = x_key.R*pvec_orig + x_key.p;

    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_size;
      if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->vec_orig[fnum].push_back(pvec_orig);
      iter->second->sig_orig[fnum].push(pvec_orig);
    }
    else
    {
      OCTO_TREE_ROOT *ot = new OCTO_TREE_ROOT(win_size, eigen_ratio);
      ot->vec_orig[fnum].push_back(pvec_orig);
      ot->sig_orig[fnum].push(pvec_orig);

      ot->voxel_center[0] = (0.5+position.x) * voxel_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->layer = 0;
      feat_map[position] = ot;
    }
  }
}

#endif