#ifndef TICK_OPTIM_PROX_SRC_PROX_H_
#define TICK_OPTIM_PROX_SRC_PROX_H_
#include <memory>
#include "base.h"
#include <string>

// TODO: all prox have a positive flag.
// TODO: use of final keyword
class Prox {
 protected:
    //! @brief Weight of the proximal operator
    double strength;

    //! @brief Flag to know if proximal operator concerns only a part of the vector
    bool has_range;

    //! @brief If range is restricted it will be applied from index start to index end
    ulong start, end;

    //! @brief If true, we apply on non negativity constraint
    bool positive;

 public:
    Prox(double strength, bool positive);

    Prox(double strength, ulong start, ulong end, bool positive);

    virtual const std::string get_class_name() const;

    virtual const bool is_separable() const;

    //! @brief call prox on coeffs, with a given step and store result in out
    virtual void call(const ArrayDouble &coeffs, double step, ArrayDouble &out);

    // TODO: remove and implement it only in prox separable?
    //! @brief call prox on coeffs, with a vector of different steps and store result in out
    virtual void call(const ArrayDouble &coeffs,
                      const ArrayDouble &step,
                      ArrayDouble &out);

    //! @brief call prox on a part of coeffs (defined by start-end), with a given step and
    //! store result in out
    virtual void call(const ArrayDouble &coeffs,
                      double step,
                      ArrayDouble &out,
                      ulong start,
                      ulong end);

    // TODO: remove and implement it only in prox separable?
    //! @brief call prox on a part of coeffs (defined by start-end), with a vector of different
    //! steps and store result in out
    virtual void call(const ArrayDouble &coeffs,
                      const ArrayDouble &step,
                      ArrayDouble &out,
                      ulong start,
                      ulong end);

    // TODO: remove and implement it only in prox separable?
    //! @brief apply prox on a single value
    virtual double call(double x,
                        double step) const;

    // TODO: remove and implement it only in prox separable?
    //! @brief apply prox on a single value several times
    virtual double call(double x,
                        double step,
                        ulong n_times) const;

    // TODO: remove and implement it only in prox separable?
    //! @brief apply prox on a single value defined by coordinate i
    virtual void call(ulong i,
                      const ArrayDouble &coeffs,
                      double step,
                      ArrayDouble &out) const;

    //! @brief apply prox on a single value defined by coordinate i several times
    virtual void call(ulong i,
                      const ArrayDouble &coeffs,
                      double step,
                      ArrayDouble &out,
                      ulong n_times) const;

    //! @brief get penalization value of the prox on the coeffs vector.
    //! This takes strength into account
    virtual double value(const ArrayDouble &coeffs);

    //! @brief get penalization value of the prox on a part of coeffs (defined by start-end).
    //! This takes strength into account
    virtual double value(const ArrayDouble &coeffs,
                         ulong start,
                         ulong end);

    //! @brief get penalization value of the prox on a single value
    //! @warning This does not take strength into account
    virtual double value(double x) const;

    //! @brief get penalization value of the prox on a single value defined by coordinate i
    //! @warning This does not take strength into account
    virtual double value(ulong i,
                         const ArrayDouble &coeffs) const;

    virtual double get_strength() const;

    virtual void set_strength(double strength);

    virtual ulong get_start() const;

    virtual ulong get_end() const;

    virtual void set_start_end(ulong start,
                               ulong end);

    virtual bool get_positive() const;

    virtual void set_positive(bool positive);
};

typedef std::shared_ptr<Prox> ProxPtr;

#endif  // TICK_OPTIM_PROX_SRC_PROX_H_
